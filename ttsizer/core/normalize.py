import json
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Optional, List, Tuple
import yaml
import os
from tqdm import tqdm
from ttsizer.utils.logger import get_logger

logger = get_logger("normalizer")

class AudioNormalizer:
    """Normalizes audio files to target loudness and true peak levels."""
    def __init__(self, config: Dict[str, Any]):
        self.lufs = config["target_lufs"]
        self.tp = config["target_tp"]
        self.sr = config["sample_rate"]
        self.codec = config.get("output_codec", "flac")
        self.timeout = config["ffmpeg_timeout_seconds"]
        self.procs = config.get("num_processes")
        self.skip_exist = config.get("skip_if_output_exists", True)

    def _run_ffmpeg(self, cmd: List[str], fname: str) -> subprocess.CompletedProcess:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=self.timeout
        )
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=result.stdout, stderr=result.stderr
            )
        return result

    def _get_loudness(self, path: Path) -> Dict[str, Any]:
        filter_str = f"loudnorm=I={self.lufs}:TP={self.tp}:print_format=json"
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
            "-af", filter_str,
            "-f", "null", "-"
        ]
        completed = self._run_ffmpeg(cmd, path.name)
        stderr = completed.stderr
        start = stderr.rfind('{')
        end = stderr.rfind('}') + 1
        if start == -1 or end == 0 or end <= start:
            raise ValueError(f"Invalid JSON in ffmpeg output for {path.name}")
        
        return json.loads(stderr[start:end])

    def _normalize(self, in_path: Path, stats: Dict[str, Any], out_path: Path):
        i = stats['input_i']
        lra = stats['input_lra']
        tp = stats['input_tp']
        thresh = stats['input_thresh']
        offset = stats.get('target_offset', 0.0)

        filter_str = (
            f"loudnorm=I={self.lufs}:LRA={lra}:TP={self.tp}:"
            f"measured_I={i}:measured_LRA={lra}:measured_TP={tp}:"
            f"measured_thresh={thresh}:offset={float(offset)}:linear=true:print_format=summary"
        )
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(in_path),
            "-af", filter_str,
            "-ac", "1",
            "-ar", str(self.sr),
            "-c:a", self.codec,
            str(out_path)
        ]
        self._run_ffmpeg(cmd, in_path.name)

    def _process_file(self, in_path: Path, out_path: Path) -> Tuple[Path, Optional[str]]:
        if self.skip_exist and out_path.exists():
            return in_path, "skipped_exists"
            
        try:
            stats = self._get_loudness(in_path)
            self._normalize(in_path, stats, out_path)
            return in_path, None
        except Exception as e:
            return in_path, str(e)

    def run(self, in_dir: Path, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing audio from {in_dir.resolve()}")
        logger.info(f"Outputting to {out_dir.resolve()}")
        if self.skip_exist:
            logger.info("Skipping existing files")

        exts = ["*.flac", "*.wav", "*.mp3", "*.aac", "*.m4a"]
        files = []
        for ext in exts:
            files.extend(list(in_dir.rglob(ext)))
        files = sorted(list(set(files)))

        if not files:
            logger.warning(f"No audio files found in {in_dir}")
            return

        tasks = []
        for in_path in files:
            rel_path = in_path.relative_to(in_dir)
            out_path = out_dir / rel_path.with_suffix(f".{self.codec}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((in_path, out_path))

        procs = self.procs if self.procs is not None else cpu_count()
        procs = max(1, min(procs, len(tasks)))

        logger.info(f"Found {len(tasks)} files. Using {procs} workers")

        success = skipped = 0
        failed = []

        with Pool(processes=procs) as pool:
            results = pool.starmap(self._process_file, tasks)
            for path, err in tqdm(results, total=len(tasks), desc="Normalizing", unit="file"):
                if err is None:
                    success += 1
                elif err == "skipped_exists":
                    skipped += 1
                else:
                    failed.append((path, err))

        logger.info(f"\nFinished processing {in_dir.name}")
        logger.info(f"Success: {success} files")
        if skipped > 0:
            logger.info(f"Skipped: {skipped} files")
        if failed:
            logger.info(f"Failed: {len(failed)} files")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    os.chdir(Path("configs/config.yaml").parent)
    
    base = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["project_name"]
    in_dir = base / cfg["vocal_removal_config"]["output_stage_folder_name"] / "vocals"
    out_dir = base / cfg["normalize_audio_config"]["output_stage_folder_name"]
    
    print(f"Testing normalization for {in_dir}")
    normalizer = AudioNormalizer(cfg["normalize_audio_config"])
    normalizer.run(in_dir, out_dir)