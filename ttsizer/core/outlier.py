import warnings
warnings.filterwarnings("ignore")

import os
import torch
import soundfile as sf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import torchaudio.functional as F
import wespeaker
import shutil
import tempfile
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from ttsizer.utils.logger import get_logger

logger = get_logger("outlier_detector")

class OutlierDetector:
    """Detects and moves audio clips that deviate from target speaker's voice profile."""

    def __init__(self, global_config: Dict[str, Any], outlier_config: Dict[str, Any]):
        model_path = global_config['global_model_paths']['speaker_embedding_model']
        self.model_path = Path(model_path).resolve()
        
        self.sr = outlier_config["target_sample_rate"]
        self.use_gpu = outlier_config["use_gpu"]
        self.min_dur = outlier_config["min_clip_duration_seconds"]
        self.refine_pct = outlier_config["centroid_refinement_percentile"]
        self.min_refine = outlier_config["min_segments_for_refinement"]
        self.min_profile = outlier_config["min_segments_for_master_profile"]
        
        self.def_thresh = outlier_config["outlier_threshold_definite"]
        self.unc_thresh = outlier_config["outlier_threshold_uncertain"]
        self.move_unc = outlier_config["move_uncertain_files"]
        self.unc_dir = outlier_config["uncertain_folder_name"]
        self.max_out_pct = outlier_config["max_outlier_percentage_warn"]
        
        self.fmt = outlier_config.get("audio_file_format_glob", "*.wav")
        self.skip_pats = outlier_config.get("skip_episode_patterns", [])
        
        self.model = self._load_model()

    def _load_model(self) -> Optional[wespeaker.Speaker]:
        model = wespeaker.load_model_local(str(self.model_path))
        model.set_device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        return model

    def _should_skip(self, ep_name: str) -> bool:
        return any(pat in ep_name for pat in self.skip_pats)

    def _get_embedding(self, path: Path) -> Optional[np.ndarray]:
        info = sf.info(str(path))
        if info.duration < self.min_dur:
            return None
        
        sig, sr = sf.read(str(path), dtype='float32')
        sig = torch.from_numpy(np.asarray(sig))

        if sig.ndim > 1 and sig.shape[-1] > 1:
            sig = torch.mean(sig, dim=-1)
        if sig.ndim == 1:
            sig = sig.unsqueeze(0)

        path_for_emb = str(path)
        tmp_file = None

        if sr != self.sr:
            sig = F.resample(sig, orig_freq=sr, new_freq=self.sr)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_file = Path(f.name)
            sf.write(str(tmp_file), sig.squeeze().cpu().numpy(), self.sr, subtype='PCM_16')
            path_for_emb = str(tmp_file)
        
        emb = self.model.extract_embedding(path_for_emb)
        
        if tmp_file and tmp_file.exists():
            tmp_file.unlink()

        if emb is None or emb.size == 0 or np.isnan(emb).any():
            return None
        return emb.flatten()

    def _setup_dirs(self, proj_name: str, src_cfg: Dict[str, Any], in_dir: Path, out_dir: Path) -> Dict[str, Dict[str, Any]]:
        src_path = Path(src_cfg["input_subpath_relative_to_input_stage"])
        spkr = src_cfg["target_speaker_label"]
        src_dir = in_dir / src_path
        out_base = out_dir / src_path
        
        if not src_dir.is_dir():
            return {}

        ep_data = {}
        for ep_dir in src_dir.iterdir():
            if not ep_dir.is_dir() or self._should_skip(ep_dir.name):
                continue

            out_spkr_dir = out_base / ep_dir.name
            filt_dir = out_spkr_dir / "filtered"
            out_dir = out_spkr_dir / "outliers"
            
            filt_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            if self.move_unc:
                (out_spkr_dir / self.unc_dir).mkdir(parents=True, exist_ok=True)

            audio_files = []
            files = list(ep_dir.glob(f"*{self.fmt[-4:]}"))
            files.extend(list(ep_dir.glob("*.txt")))

            if not files:
                continue
            
            for src in files:
                dst = filt_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                if src.suffix.lower() == self.fmt[-4:].lower():
                    audio_files.append(dst)

            if audio_files:
                ep_data[ep_dir.name] = {
                    'spkr_dir': out_spkr_dir,
                    'filt_dir': filt_dir,
                    'audio_files': sorted(audio_files)
                }
        return ep_data

    def _get_profile(self, files: List[Path], name: str) -> Tuple[Optional[np.ndarray], Dict[Path, np.ndarray]]:
        embs = {}
        emb_list = []

        for path in tqdm(files, desc=f"Embedding {name}", leave=False):
            emb = self._get_embedding(path)
            if emb is not None:
                embs[path] = emb
                emb_list.append(emb)

        if len(emb_list) < self.min_profile:
            return None, embs

        if not all(emb.shape == emb_list[0].shape and len(emb.shape) == 1 for emb in emb_list):
            return None, embs

        emb_arr = np.array(emb_list)
        centroid = np.mean(emb_arr, axis=0)

        if emb_arr.shape[0] >= self.min_refine:
            dists = 1.0 - cosine_similarity(centroid.reshape(1, -1), emb_arr)[0]
            if len(dists) > 0:
                thresh = np.percentile(dists, self.refine_pct)
                idx = np.where(dists <= thresh)[0]
                if len(idx) >= self.min_refine:
                    centroid = np.mean(emb_arr[idx], axis=0)
            
        return centroid, embs

    def _process_episode(
        self, ep_name: str, ep_dir: Path, 
        files: List[Dict[str, Any]], centroid: np.ndarray
    ) -> Tuple[Dict[str, List[Dict]], int, int]:
        
        filt_dir = ep_dir / "filtered"
        out_dir = ep_dir / "outliers"
        unc_dir = ep_dir / self.unc_dir

        def_moved = unc_moved = 0
        def_report = []
        unc_report = []
        
        unc_thresh = min(self.unc_thresh, self.def_thresh - 0.001)
        centroid_2d = centroid.reshape(1, -1)

        for item in files:
            path = item['path']
            emb = item['embedding'].reshape(1, -1)
            
            if not path.exists():
                continue

            dist = 1.0 - cosine_similarity(centroid_2d, emb)[0][0]
            move_dir = None
            report = None
            is_def = is_unc = False

            if dist > self.def_thresh:
                move_dir, report, is_def = out_dir, def_report, True
            elif dist > unc_thresh:
                report = unc_report
                if self.move_unc:
                    move_dir, is_unc = unc_dir, True
            
            if move_dir:
                base = path.name
                txt = path.with_suffix(".txt").name
                txt_src = filt_dir / txt
                
                try:
                    shutil.move(str(path), str(move_dir / base))
                    if txt_src.exists():
                        shutil.move(str(txt_src), str(move_dir / txt))
                    
                    if is_def:
                        def_moved += 1
                    elif is_unc:
                        unc_moved += 1
                        
                    report.append({
                        'file': base,
                        'distance': dist,
                        'threshold': self.def_thresh if is_def else unc_thresh
                    })
                except Exception:
                    continue

        return {
            'definite': def_report,
            'uncertain': unc_report
        }, def_moved, unc_moved

    def _print_summary(self, src_name: str, out_base: Path, def_moved: int, unc_moved: int, 
                      reports: Dict[str, Dict[str, List[Dict]]], details: Dict[str, Dict[str, Any]]):
        total_files = sum(len(d['audio_files']) for d in details.values())
        if total_files == 0:
            return

        def_pct = (def_moved / total_files) * 100
        unc_pct = (unc_moved / total_files) * 100
        
        logger.info(f"\n=== Summary for {src_name} ===")
        logger.info(f"Total files processed: {total_files}")
        logger.info(f"Definite outliers: {def_moved} ({def_pct:.1f}%)")
        logger.info(f"Uncertain files: {unc_moved} ({unc_pct:.1f}%)")
        
        if def_pct > self.max_out_pct:
            logger.warning(f"High outlier rate ({def_pct:.1f}%) exceeds warning threshold ({self.max_out_pct}%)")

    def run(self, proj_name: str, srcs: List[Dict[str, Any]], in_dir: Path, out_dir: Path):
        for src in srcs:
            src_name = src["target_speaker_label"]
            ep_data = self._setup_dirs(proj_name, src, in_dir, out_dir)
            
            if not ep_data:
                continue

            all_files = []
            for ep_name, data in ep_data.items():
                all_files.extend(data['audio_files'])

            centroid, embs = self._get_profile(all_files, src_name)
            if centroid is None:
                continue

            def_moved = unc_moved = 0
            reports = {}

            for ep_name, data in ep_data.items():
                files = [{'path': p, 'embedding': embs[p]} for p in data['audio_files'] if p in embs]
                if not files:
                    continue

                ep_reports, def_cnt, unc_cnt = self._process_episode(
                    ep_name, data['spkr_dir'], files, centroid
                )
                
                def_moved += def_cnt
                unc_moved += unc_cnt
                reports[ep_name] = ep_reports

            self._print_summary(
                src_name, 
                out_dir / Path(src["input_subpath_relative_to_input_stage"]),
                def_moved, unc_moved, reports, ep_data
            )


if __name__ == '__main__':
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    os.chdir(Path("configs/config.yaml").parent)
    
    in_dir = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["project_name"] / cfg["extract_audio_config"]["output_stage_folder_name"]
    out_dir = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["project_name"] / cfg["outlier_detection_config"]["output_stage_folder_name"]
    
    print(f"Testing outlier detection for {in_dir}")
    detector = OutlierDetector(global_config=cfg, outlier_config=cfg["outlier_detection_config"])
    detector.run(cfg["project_setup"]["project_name"], cfg["outlier_detection_config"]["audio_sources"], in_dir, out_dir) 