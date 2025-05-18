import subprocess
import json
from tqdm import tqdm
from pathlib import Path
import tempfile
from typing import Optional, List, Dict, Any
import multiprocessing
import yaml # For loading config in __main__
import os # For os.chdir in __main__

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
        self.preferred_lang_codes: List[str] = config["preferred_lang_codes"]
        self.max_workers: Optional[int] = config.get("max_workers") # Handled by multiprocessing.Pool if None
        self.output_sample_rate: str = str(config["output_sample_rate"])
        self.output_codec: str = config["output_codec"]
        self.resolution_threshold_aac: int = config["resolution_threshold_for_aac_step"]
        self.intermediate_aac_bitrate: str = config["intermediate_aac_bitrate"]
        # Note: tqdm.write is used for logging from workers, which is fine for console.
        # For file logging from workers, a more complex setup (e.g., queue handler) would be needed.

    def _probe_audio_streams(self, input_path: Path) -> List[Dict[str, Any]]:
        # Probes a video file for all its audio streams.
        cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "a", str(input_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout).get("streams", [])
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            tqdm.write(f"❌ ffprobe (audio) error for {input_path.name}: {type(e).__name__}")
        return []

    def _probe_video_height(self, input_path: Path) -> Optional[int]:
        # Probes a video file for its video height.
        cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "v:0", str(input_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            streams = json.loads(result.stdout).get("streams", [])
            if streams and "height" in streams[0]:
                return int(streams[0]["height"])
        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError, TypeError, IndexError):
            # Silently fail if height cannot be determined, or ffprobe error.
            pass 
        return None

    def _select_audio_stream_index(self, streams: List[Dict[str, Any]]) -> Optional[int]:
        # Selects the most preferred audio stream based on language tags.
        for code in self.preferred_lang_codes:
            for s in streams:
                if s.get("tags", {}).get("language", "").lower() == code.lower():
                    return s.get("index")
        return streams[0].get("index") if streams else None

    def _extract_and_convert_single_file(self, input_video_path: Path, output_audio_path: Path) -> bool:
        # Worker function to extract audio from a single video file.
        audio_streams = self._probe_audio_streams(input_video_path)
        audio_stream_idx = self._select_audio_stream_index(audio_streams)

        if audio_stream_idx is None:
            tqdm.write(f"❌ No suitable audio stream in {input_video_path.name}. Skipping.")
            return False

        audio_map_spec = f"0:{audio_stream_idx}"
        video_height = self._probe_video_height(input_video_path)
        perform_aac_step = bool(video_height and video_height > self.resolution_threshold_aac)

        common_ffmpeg_params = ["-y", "-hide_banner", "-nostats", "-i", str(input_video_path), "-map", audio_map_spec]
        
        try:
            if perform_aac_step:
                with tempfile.TemporaryDirectory(prefix="audio_conv_") as temp_dir_name:
                    temp_aac_path = Path(temp_dir_name) / f"{input_video_path.stem}_temp.aac"
                    cmd_to_aac = ["ffmpeg", *common_ffmpeg_params[1:], "-c:a", "aac",
                                  "-b:a", self.intermediate_aac_bitrate, "-ar", self.output_sample_rate, str(temp_aac_path)]
                    subprocess.run(cmd_to_aac, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    cmd_to_final = ["ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(temp_aac_path),
                                    "-c:a", self.output_codec, "-ar", self.output_sample_rate, str(output_audio_path)]
                    subprocess.run(cmd_to_final, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # tqdm.write(f"✅ Converted {input_video_path.name} (via AAC)") # Optional: for less verbose logs
            else:
                cmd_direct_to_final = ["ffmpeg", *common_ffmpeg_params[1:], "-c:a", self.output_codec,
                                       "-ar", self.output_sample_rate, str(output_audio_path)]
                subprocess.run(cmd_direct_to_final, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # tqdm.write(f"✅ Converted {input_video_path.name} (direct)") # Optional
            return True
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode(errors="ignore").strip()
            last_line = err_msg.splitlines()[-1] if err_msg.splitlines() else 'FFmpeg error'
            tqdm.write(f"❌ Conversion Failed for {input_video_path.name}: {last_line}")
            return False
        except Exception as e_gen:
            tqdm.write(f"❌ Unexpected error for {input_video_path.name}: {e_gen}")
            return False

    def run_extraction_for_project(self, input_project_video_dir: Path, output_project_audio_base_dir: Path):
        """
        Extracts audio for all videos in a project directory.
        Args:
            input_project_video_dir: Path to the directory containing video files for the project.
            output_project_audio_base_dir: Path to the base directory where extracted audio
                                           (e.g., .../project_name/01_extracted_audio) will be stored.
                                           An 'orig' subfolder will be created here.
        """
        output_target_main_dir = output_project_audio_base_dir / "orig"
        output_target_main_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nAudioExtraction: Processing videos from {input_project_video_dir}")
        print(f"AudioExtraction: Outputting to {output_target_main_dir}")
        
        workers_to_use = self.max_workers if self.max_workers is not None else multiprocessing.cpu_count()
        workers_to_use = max(1, workers_to_use)

        video_exts = {'.mkv', '.mp4', '.avi', '.mov', '.webm'}
        tasks = []
        for f_path in input_project_video_dir.rglob("*"):
            if f_path.is_file() and f_path.suffix.lower() in video_exts:
                relative_path = f_path.relative_to(input_project_video_dir)
                output_file_subdir = output_target_main_dir / relative_path.parent
                output_file_subdir.mkdir(parents=True, exist_ok=True)
                out_audio_path = output_file_subdir / f_path.with_suffix(f'.{self.output_codec.lower()}').name
                tasks.append((f_path, out_audio_path))

        if not tasks:
            print(f"AudioExtraction: No video files found in {input_project_video_dir}.")
            return # Corrected: return from function
    
        workers_to_use = min(workers_to_use, len(tasks))
        
        print(f"AudioExtraction: Found {len(tasks)} videos. Starting extraction with {workers_to_use} workers...")
        with multiprocessing.Pool(processes=workers_to_use) as pool:
            results = list(tqdm(pool.starmap(self._extract_and_convert_single_file, tasks),
                            total=len(tasks),
                                desc=f"Extracting audio: {input_project_video_dir.name}",
                                unit="file"))
        
        successful = sum(1 for r in results if r is True)
        print(f"\nAudioExtraction: Finished for {input_project_video_dir.name}. Success: {successful}/{len(tasks)} files.")


# For standalone testing:
if __name__ == '__main__':
    multiprocessing.freeze_support() # Ensure this is at the top for multiprocessing
    
    # Config loading assumes config.yaml is in the current or parent directory if script is in a subdir
    config_file = Path("config.yaml")
    if not config_file.exists():
        config_file = Path("../config.yaml") # Simple check for one level up
    
    if not config_file.exists(): # Check again after trying one level up
        print("ERROR: config.yaml not found in . or ../ for standalone test.")
        exit(1)
            
    cfg = None
    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Could not load or parse {config_file.resolve()}: {e}")
        exit(1)
        
    if cfg is None: # Should not happen if try/except is correct, but as a safeguard
        print("ERROR: Failed to load cfg, it is None.")
        exit(1)

    # Set CWD to project root (where config.yaml is) for consistent relative path resolution
    project_root_for_test = config_file.parent.resolve()
    if Path.cwd() != project_root_for_test:
        print(f"Standalone test: Changing CWD to project root: {project_root_for_test}")
        os.chdir(project_root_for_test)

    project_setup = cfg.get("project_setup")
    if not project_setup:
        print("ERROR: 'project_setup' section not found in config.yaml.")
        exit(1)

    extractor_cfg = cfg.get("extract_audio_config")
    if not extractor_cfg:
        print("ERROR: 'extract_audio_config' section not found in config.yaml.")
        exit(1)
    
    project_name = project_setup.get("project_name")
    if not project_name:
        print("ERROR: 'project_name' not found in project_setup in config.yaml")
        exit(1)
        
    video_source_parent_str = project_setup.get("video_source_parent_dir")
    if not video_source_parent_str:
        print("ERROR: 'video_source_parent_dir' not found in project_setup in config.yaml")
        exit(1)
    video_source_parent_abs = Path(video_source_parent_str)

    output_base_str = project_setup.get("output_base_dir")
    if not output_base_str:
        print("ERROR: 'output_base_dir' not found in project_setup in config.yaml")
        exit(1)
    output_base_abs = Path(output_base_str)

    if not video_source_parent_abs.is_absolute():
        video_source_parent_abs = (project_root_for_test / video_source_parent_abs).resolve()
    if not output_base_abs.is_absolute():
        output_base_abs = (project_root_for_test / output_base_abs).resolve()

    input_videos_project_dir = video_source_parent_abs / project_name
    
    stage_output_folder = extractor_cfg.get("output_stage_folder_name", "01_extracted_audio_STANDALONE_TEST")
    output_audio_base_for_stage = output_base_abs / project_name / stage_output_folder

    if not input_videos_project_dir.is_dir():
        print(f"ERROR: Test input video directory not found: {input_videos_project_dir}")
        exit(1)
            
    print(f"--- Running AudioExtractor Standalone Test for project: {project_name} ---")
    print(f"  Input Videos from: {input_videos_project_dir.relative_to(project_root_for_test) if input_videos_project_dir.is_relative_to(project_root_for_test) else input_videos_project_dir}")
    print(f"  Output Base to: {output_audio_base_for_stage.relative_to(project_root_for_test) if output_audio_base_for_stage.is_relative_to(project_root_for_test) else output_audio_base_for_stage} (it will create 'orig' inside)")
    
    extractor = AudioExtractor(config=extractor_cfg)
    extractor.run_extraction_for_project(
        input_project_video_dir=input_videos_project_dir,
        output_project_audio_base_dir=output_audio_base_for_stage
    )
    print(f"--- Standalone Test Finished. Check output in: {output_audio_base_for_stage / 'orig'} ---")