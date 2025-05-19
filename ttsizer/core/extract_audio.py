import subprocess
import json
from tqdm import tqdm
from pathlib import Path
import tempfile
from typing import Optional, List, Dict, Any
import multiprocessing
import yaml
import os
from ttsizer.utils.logger import get_logger

logger = get_logger("audio_extractor")

class AudioExtractor:
    """
    Extracts audio from video files based on configuration settings.
    All configurations are passed via a dictionary during instantiation.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AudioExtractor with specific configuration.
        Args:
            config: A dictionary containing settings for audio extraction,
                    typically from 'extract_audio_config' in config.yaml.
        """
        self.lang_codes = config["preferred_lang_codes"]
        self.workers = config.get("max_workers")
        self.sample_rate = str(config["output_sample_rate"])
        self.codec = config["output_codec"]
        self.aac_threshold = config["resolution_threshold_for_aac_step"]
        self.aac_bitrate = config["intermediate_aac_bitrate"]

    def _get_audio_streams(self, path: Path) -> List[Dict[str, Any]]:
        cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "a", str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout).get("streams", [])

    def _get_video_height(self, path: Path) -> Optional[int]:
        cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "v:0", str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        streams = json.loads(result.stdout).get("streams", [])
        return int(streams[0]["height"]) if streams and "height" in streams[0] else None

    def _get_stream_idx(self, streams: List[Dict[str, Any]]) -> Optional[int]:
        for code in self.lang_codes:
            for s in streams:
                if s.get("tags", {}).get("language", "").lower() == code.lower():
                    return s.get("index")
        return streams[0].get("index") if streams else None

    def _process_file(self, in_path: Path, out_path: Path) -> bool:
        streams = self._get_audio_streams(in_path)
        stream_idx = self._get_stream_idx(streams)

        if stream_idx is None:
            tqdm.write(f"❌ No audio stream in {in_path.name}")
            return False

        audio_map = f"0:{stream_idx}"
        height = self._get_video_height(in_path)
        use_aac = bool(height and height > self.aac_threshold)

        ffmpeg_base = ["-y", "-hide_banner", "-nostats", "-i", str(in_path), "-map", audio_map]
        
        try:
            if use_aac:
                with tempfile.TemporaryDirectory(prefix="tmp_") as tmp_dir:
                    tmp_path = Path(tmp_dir) / f"{in_path.stem}_tmp.aac"
                    cmd_aac = ["ffmpeg", *ffmpeg_base[1:], "-c:a", "aac",
                             "-b:a", self.aac_bitrate, "-ar", self.sample_rate, str(tmp_path)]
                    subprocess.run(cmd_aac, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    cmd_final = ["ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(tmp_path),
                               "-c:a", self.codec, "-ar", self.sample_rate, str(out_path)]
                    subprocess.run(cmd_final, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                cmd = ["ffmpeg", *ffmpeg_base[1:], "-c:a", self.codec,
                      "-ar", self.sample_rate, str(out_path)]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="ignore").strip()
            last_line = err.splitlines()[-1] if err.splitlines() else 'FFmpeg error'
            tqdm.write(f"❌ Failed {in_path.name}: {last_line}")
            return False

    def run_extraction_for_project(self, in_dir: Path, out_dir: Path):
        """
        Extracts audio for all videos in a project directory.
        Args:
            in_dir: Path to the directory containing video files for the project.
            out_dir: Path to the base directory where extracted audio
                                           (e.g., .../project_name/01_extracted_audio) will be stored.
                                           An 'orig' subfolder will be created here.
        """
        out_dir = out_dir / "orig"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing: {in_dir}")
        logger.info(f"Output: {out_dir}")
        
        workers = self.workers if self.workers is not None else multiprocessing.cpu_count()
        workers = max(1, workers)

        video_exts = {'.mkv', '.mp4', '.avi', '.mov', '.webm'}
        tasks = []
        for path in in_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in video_exts:
                rel_path = path.relative_to(in_dir)
                out_subdir = out_dir / rel_path.parent
                out_subdir.mkdir(parents=True, exist_ok=True)
                out_path = out_subdir / path.with_suffix(f'.{self.codec.lower()}').name
                tasks.append((path, out_path))

        if not tasks:
            logger.warning(f"No videos found in {in_dir}")
            return
    
        workers = min(workers, len(tasks))
        
        logger.info(f"Found {len(tasks)} videos. Using {workers} workers...")
        with multiprocessing.Pool(processes=workers) as pool:
            results = list(tqdm(pool.starmap(self._process_file, tasks),
                            total=len(tasks),
                            desc=f"Extracting: {in_dir.name}",
                            unit="file"))
        
        success = sum(1 for r in results if r is True)
        logger.info(f"Done: {success}/{len(tasks)} files processed")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    os.chdir(Path("configs/config.yaml").parent)
    
    in_dir = Path(cfg["project_setup"]["video_source_parent_dir"]) / cfg["project_setup"]["project_name"]
    out_dir = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["project_name"] / cfg["extract_audio_config"]["output_stage_folder_name"]
    
    print(f"Testing extraction for {in_dir}")
    extractor = AudioExtractor(config=cfg["extract_audio_config"])
    extractor.run_extraction_for_project(in_dir, out_dir)