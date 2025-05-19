import warnings
warnings.filterwarnings("ignore")

import os
import nemo.collections.asr as nemo_asr
import soundfile as sf
import torch
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import shutil
import yaml
from typing import Dict, Any, List, Optional, Tuple
from ttsizer.utils.logger import get_logger

logger = get_logger("parakeet")

class ParakeetASRProcessor:
    """Processes audio files using a Parakeet ASR model for transcription and flagging."""

    def __init__(self, global_config: Dict[str, Any], parakeet_config: Dict[str, Any]):
        """
        Initializes the ParakeetASRProcessor with settings from configuration.
        Args:
            global_config: Contains global project settings. CWD is assumed to be project root.
            parakeet_config: Contains settings specific to Parakeet ASR processing.
        """
        # ASR model and processing parameters
        self.model_name = parakeet_config["asr_model_name"]
        self.batch_size = parakeet_config["batch_size"]
        self.device = parakeet_config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Timestamp and flagging parameters
        self.time_thresh = parakeet_config["timestamp_deviation_threshold_sec"]
        self.padding = parakeet_config["padding_sec"]

        # Output subfolder names from config
        self.flagged_dir = parakeet_config["output_subfolder_names"]["flagged_cropped"]
        self.fmt = parakeet_config.get("audio_file_format_glob", "*.wav")

        self.model = None
        self._load_model()
        logger.info("ParakeetASRProcessor initialized.")

    def _load_model(self):
        logger.info(f"Loading ASR model: {self.model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name, 
            map_location=self.device
        )
        self.model.eval()

    def _get_times(self, result: Any, dur: float) -> Tuple[float, float]:
        start, end = 0.0, dur
        if hasattr(result, 'timestamp') and isinstance(result.timestamp, dict) and 'word' in result.timestamp:
            words = result.timestamp['word']
            valid = [s for s in words if isinstance(s, dict) and 'start' in s and 'end' in s]
            if valid:
                start = max(0.0, min(s['start'] for s in valid))
                end = min(dur, max(s['end'] for s in valid))
                if end < start:
                    start, end = 0.0, dur
        return start, end

    def run(self, in_dir: Path, out_dir: Path):
        if not self.model:
            logger.error("ASR model not loaded")
            return

        flagged_dir = out_dir / self.flagged_dir
        if flagged_dir.exists():
            shutil.rmtree(flagged_dir)
        flagged_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Input: {in_dir.resolve()}")
        logger.info(f"Flagged output: {flagged_dir.resolve()}")

        files = sorted(list(in_dir.rglob(self.fmt)))
        if not files:
            logger.warning(f"No '{self.fmt}' files found in {in_dir}")
            return

        logger.info(f"Found {len(files)} files")
        flagged = 0

        for i in tqdm(range(0, len(files), self.batch_size), desc="Processing", unit="batch"):
            batch = [str(p) for p in files[i:i + self.batch_size]]
            batch_files = files[i:i + self.batch_size]

            try:
                results = self.model.transcribe(
                    batch,
                    batch_size=len(batch),
                    timestamps=True,
                    verbose=False
                )
            except Exception as e:
                logger.error(f"Error in batch: {e}")
                continue

            if not results or len(results) != len(batch_files):
                continue

            if isinstance(results, tuple) and len(results) == 1 and isinstance(results[0], list):
                results = results[0]
            elif not isinstance(results, list):
                continue

            if len(results) != len(batch_files):
                continue

            for path, result in zip(batch_files, results):
                try:
                    audio, sr = sf.read(str(path), dtype='float32')
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1)
                    dur = len(audio) / sr if sr > 0 else 0.0
                    
                    text = ""
                    start, end = 0.0, dur

                    if isinstance(result, str):
                        text = result
                    elif hasattr(result, 'text'):
                        text = result.text or ""
                        start, end = self._get_times(result, dur)

                    # Check if needs flagging
                    start_dev = abs(start - 0.0)
                    end_dev = abs(end - dur)
                    
                    if start_dev > self.time_thresh or end_dev > self.time_thresh:
                        pad_start = max(0.0, start - self.padding)
                        pad_end = min(dur, end + self.padding)
                        
                        if pad_end <= pad_start:
                            if end > start:
                                pad_start, pad_end = start, end
                            else:
                                pad_start, pad_end = 0.0, dur

                        start_frame = int(pad_start * sr)
                        end_frame = int(pad_end * sr)

                        crop_audio = audio
                        if dur > 0 and end_frame > start_frame and start_frame >= 0 and end_frame <= len(audio):
                            crop_audio = audio[start_frame:end_frame]
                        
                        if crop_audio.size == 0:
                            if audio.size > 0:
                                crop_audio = audio
                            else:
                                continue
                        
                        flagged_audio = flagged_dir / path.name
                        flagged_txt = flagged_dir / path.with_suffix(".txt").name
                        sf.write(str(flagged_audio), crop_audio, sr, subtype='PCM_24')
                        with open(flagged_txt, 'w', encoding='utf-8') as f:
                            f.write(text)
                        flagged += 1
                
                except Exception as e:
                    logger.error(f"Error processing {path.name}: {e}")

        logger.info("\nProcessing complete")
        logger.info(f"Total files: {len(files)}")
        logger.info(f"Flagged: {flagged}")

if __name__ == "__main__":
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        cfg_path = Path("../config.yaml")
    
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        exit(1)

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    root = cfg_path.parent
    if Path.cwd() != root:
        os.chdir(root)

    setup = cfg["project_setup"]
    name = setup["project_name"]
    out_dir = Path(setup["output_base_dir"])
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    
    proj_dir = out_dir / name
    parakeet_cfg = cfg["parakeet_asr_config"]
    fo_cfg = cfg["find_outliers_config"]
    
    for src in fo_cfg["project_audio_sources"]:
        src_path = Path(src["input_subpath_relative_to_input_stage"])
        in_dir = proj_dir / fo_cfg["output_stage_folder_name"] / src_path / "filtered"
        out_dir = proj_dir / parakeet_cfg["output_stage_folder_name"] / src_path
        
        if not in_dir.is_dir():
            print(f"Input not found: {in_dir}")
            continue
            
        print(f"\nProcessing: {src['target_speaker_label']}")
        print(f"Input: {in_dir}")
        print(f"Output: {out_dir}")

        processor = ParakeetASRProcessor(cfg, parakeet_cfg)
        processor.run(in_dir, out_dir)
    
    print(f"\nDone. Output in: {proj_dir / parakeet_cfg['output_stage_folder_name']}")