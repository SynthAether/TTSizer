import json
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Optional, List, Tuple
import yaml # For loading config in __main__
import os # For os.chdir in __main__
from tqdm import tqdm # For progress bar

class AudioNormalizer:
    """Normalizes audio files to target loudness and true peak levels."""
    def __init__(self, config: Dict[str, Any]):
        self.target_lufs: float = config["target_lufs"]
        self.target_tp: float = config["target_tp"]
        self.sample_rate: int = config["sample_rate"]
        self.output_codec: str = config.get("output_codec", "flac") # Default to flac if not specified
        self.ffmpeg_timeout: int = config["ffmpeg_timeout_seconds"]
        self.num_processes: Optional[int] = config.get("num_processes")
        self.skip_if_output_exists: bool = config.get("skip_if_output_exists", True)
        # tqdm.write is used for logging from worker processes

    def _run_ffmpeg_command(self, cmd: List[str], file_name_for_log: str) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False, # Will check returncode manually
                timeout=self.ffmpeg_timeout
            )
        except subprocess.TimeoutExpired as e:
            # Extract last few lines of stderr if available
            stderr_tail = ""
            if e.stderr:
                try:
                    stderr_lines = e.stderr.decode(errors='replace').strip().splitlines()
                    stderr_tail = "\n".join(stderr_lines[-5:]) # Last 5 lines
                except Exception:
                    stderr_tail = "Could not decode stderr."

            error_message = (f"FFmpeg timeout for {file_name_for_log} after {self.ffmpeg_timeout}s. "
                             f"Command: '{' '.join(cmd)}'. Stderr (tail): {stderr_tail}")
            raise TimeoutError(error_message) from e

        if result.returncode != 0:
            # Extract last few lines of stderr for CalledProcessError as well
            stderr_tail = ""
            if result.stderr:
                stderr_lines = result.stderr.strip().splitlines()
                stderr_tail = "\n".join(stderr_lines[-5:])
            
            # Create a new CalledProcessError with a more informative message including the stderr tail
            # Note: We are not re-raising result.check_returncode() as it might not include stderr directly
            # in its formatted message in all Python versions in the way we want.
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=result.stdout, stderr=result.stderr,
            )
        return result

    def _measure_loudness(self, input_path: Path) -> Dict[str, Any]:
        filter_str = f"loudnorm=I={self.target_lufs}:TP={self.target_tp}:print_format=json"
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(input_path),
            "-af", filter_str,
            "-f", "null", "-"
        ]
        # tqdm.write(f"[{input_path.name}] Measuring loudness...") # Less verbose for workers
        completed = self._run_ffmpeg_command(cmd, input_path.name)
        stderr = completed.stderr
        start = stderr.rfind('{')
        end = stderr.rfind('}') + 1 # Look for last brace
        if start == -1 or end == 0 or end <= start:
            raise ValueError(f"Could not find valid JSON in ffmpeg output for {input_path.name}. Stderr: {stderr[-500:]}")
        
        json_str = stderr[start:end]
        try:
            stats = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from ffmpeg for {input_path.name}. Error: {e}. JSON: '{json_str}'. Stderr: {stderr[-500:]}")
        return stats

    def _normalize_and_convert(self, input_path: Path, stats: Dict[str, Any], output_path: Path):
        measured_I = stats['input_i']
        measured_LRA = stats['input_lra']
        measured_TP = stats['input_tp']
        measured_thresh = stats['input_thresh']
        offset = stats.get('target_offset', 0.0) # Ensure offset is correctly parsed as float

        loudnorm_filter = (
            f"loudnorm=I={self.target_lufs}:LRA={measured_LRA}:TP={self.target_tp}:"
            f"measured_I={measured_I}:measured_LRA={measured_LRA}:measured_TP={measured_TP}:"
            f"measured_thresh={measured_thresh}:offset={float(offset)}:linear=true:print_format=summary"
        )
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(input_path),
            "-af", loudnorm_filter,
            "-ac", "1", # Mono output
            "-ar", str(self.sample_rate),
            "-c:a", self.output_codec, # e.g. flac
            str(output_path)
        ]
        # tqdm.write(f"[{input_path.name}] Normalizing to {output_path.name}...") # Less verbose for workers
        self._run_ffmpeg_command(cmd, input_path.name)

    def _process_single_file_worker(self, input_path: Path, output_path: Path) -> Tuple[Path, Optional[str]]:
        # Renamed from process_single_file to avoid conflict if orchestrator calls this module directly
        # This is the function that will be mapped by multiprocessing.Pool
        if self.skip_if_output_exists and output_path.exists():
            # tqdm.write(f"Skipping {input_path.name} as output {output_path.name} already exists.")
            return input_path, "skipped_exists"
        try:
            stats = self._measure_loudness(input_path)
            self._normalize_and_convert(input_path, stats, output_path)
            # tqdm.write(f"[{input_path.name}] Successfully processed.") # Worker should not print success, main loop will
            return input_path, None # Success
        except (subprocess.CalledProcessError, TimeoutError, ValueError) as e:
            # For CalledProcessError, include stderr if available
            err_msg = str(e)
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                stderr_lines = e.stderr.strip().splitlines()
                stderr_tail = "\n".join(stderr_lines[-5:]) # Last 5 lines of stderr
                err_msg = f"{type(e).__name__} for '{input_path.name}'. Stderr (tail): {stderr_tail}. Command: '{' '.join(e.cmd)}'"
            elif isinstance(e, TimeoutError):
                 err_msg = f"Timeout for '{input_path.name}'. {e}"
            else: # ValueError
                err_msg = f"{type(e).__name__} for '{input_path.name}'. {e}"

            # tqdm.write(f"[{input_path.name}] Error: {err_msg.splitlines()[0]}...") # Log from worker
            return input_path, err_msg # Failure, return error message
        except Exception as e: # Catch any other unexpected errors
            # tqdm.write(f"[{input_path.name}] Unexpected error: {type(e).__name__}: {e}")
            return input_path, f"Unexpected error for '{input_path.name}': {type(e).__name__}: {e}"


    def run_normalization_for_project(self, project_input_dir: Path, project_output_dir: Path):
        project_output_dir.mkdir(parents=True, exist_ok=True)
        
        tqdm.write(f"\nAudioNormalizer: Processing audio from {project_input_dir.resolve()}")
        tqdm.write(f"AudioNormalizer: Outputting normalized audio to {project_output_dir.resolve()}")
        if self.skip_if_output_exists:
            tqdm.write("AudioNormalizer: Will skip files if output already exists.")

        # Glob for common audio file types, can be made configurable
        audio_extensions = ["*.flac", "*.wav", "*.mp3", "*.aac", "*.m4a"]
        files_to_process = []
        for ext in audio_extensions:
            files_to_process.extend(list(project_input_dir.rglob(ext)))
        files_to_process = sorted(list(set(files_to_process))) # Remove duplicates and sort

        if not files_to_process:
            tqdm.write(f"AudioNormalizer: No compatible audio files found in {project_input_dir}")
            return

        tasks = []
        for input_file_path in files_to_process:
            # Determine output path, maintaining relative structure if input is from subdirs
            relative_path = input_file_path.relative_to(project_input_dir)
            output_file_path = project_output_dir / relative_path.with_suffix(f".{self.output_codec}")
            output_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure subfolder exists
            tasks.append((input_file_path, output_file_path))

        workers = self.num_processes if self.num_processes is not None else cpu_count()
        workers = max(1, min(workers, len(tasks))) # Ensure at least 1 worker, and not more than tasks

        tqdm.write(f"AudioNormalizer: Found {len(tasks)} audio files. Starting normalization with {workers} workers...")
        
        successful_count = 0
        skipped_count = 0
        failed_items: List[Tuple[Path, str]] = []

        with Pool(processes=workers) as pool:
            # Using tqdm to wrap the pool.imap_unordered for progress bar
            results_iterable = pool.starmap(self._process_single_file_worker, tasks)
            for result in tqdm(results_iterable, total=len(tasks), desc=f"Normalizing Audio", unit="file"):
                file_path, error_obj = result
                if error_obj is None:
                    successful_count += 1
                elif error_obj == "skipped_exists":
                    skipped_count +=1
                else:
                    failed_items.append((file_path, error_obj))
                    # Error already logged by tqdm.write in the worker for immediate feedback

        tqdm.write(f"\nAudioNormalizer: Finished for {project_input_dir.name}.")
        tqdm.write(f"  Successfully processed: {successful_count} files.")
        if skipped_count > 0:
            tqdm.write(f"  Skipped (output existed): {skipped_count} files.")
        if failed_items:
            tqdm.write(f"  Failed to process {len(failed_items)} files (see logs above for details):")
            # for f_path, err_msg in failed_items: # No need to re-print, already logged by worker
            #     tqdm.write(f"    - {f_path.name}: {err_msg.splitlines()[0]}")


# For standalone testing:
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    config_file = Path("config.yaml")
    if not config_file.exists():
        config_file = Path("../config.yaml")
    
    if not config_file.exists():
        print("ERROR: config.yaml not found for standalone test.")
        exit(1)

    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Could not load or parse {config_file.resolve()}: {e}")
        exit(1)

    project_root_for_test = config_file.parent.resolve()
    if Path.cwd() != project_root_for_test:
        print(f"Standalone test: Changing CWD to project root: {project_root_for_test}")
        os.chdir(project_root_for_test)

    project_setup = cfg.get("project_setup")
    if not project_setup:
        print("ERROR: 'project_setup' section not found in config.yaml.")
        exit(1)

    project_name = project_setup.get("project_name")
    output_base_abs = Path(project_setup.get("output_base_dir"))
    if not output_base_abs.is_absolute():
        output_base_abs = (project_root_for_test / output_base_abs).resolve()
    
    processing_project_dir_abs = output_base_abs / project_name

    # Input for normalization is the output of vocal removal (vocal_removal_config.output_stage_folder_name / 'vocals')
    vocal_removal_cfg = cfg.get("vocal_removal_config", {})
    vocal_removal_output_folder = vocal_removal_cfg.get("output_stage_folder_name", "02_vocals_removed_test")
    input_audio_dir = processing_project_dir_abs / vocal_removal_output_folder / "vocals"

    # Output for normalization (normalize_audio_config.output_stage_folder_name)
    normalizer_stage_config = cfg.get("normalize_audio_config")
    if not normalizer_stage_config:
        print(f"ERROR: 'normalize_audio_config' not found in config.yaml.")
        exit(1)
    normalizer_output_folder = normalizer_stage_config.get("output_stage_folder_name", "03_vocals_normalized_STANDALONE_TEST")
    output_normalized_dir = processing_project_dir_abs / normalizer_output_folder

    if not input_audio_dir.is_dir():
        print(f"ERROR: Test input audio directory not found: {input_audio_dir.resolve()}")
        print(f"Ensure the output from a previous vocal removal test run exists (e.g., '{vocal_removal_output_folder}/vocals').")
        exit(1)
            
    print(f"--- Running AudioNormalizer Standalone Test for project: {project_name} ---")
    print(f"Input audio from: {input_audio_dir.resolve()}")
    print(f"Output normalized to: {output_normalized_dir.resolve()}")

    try:
        normalizer = AudioNormalizer(config=normalizer_stage_config) # Pass only its own config block
        normalizer.run_normalization_for_project(
            project_input_dir=input_audio_dir,
            project_output_dir=output_normalized_dir
        )
    except Exception as e_main:
        print(f"ERROR: An unexpected error occurred during standalone AudioNormalizer test: {e_main}")
        import traceback
        traceback.print_exc()
    
    print(f"--- Standalone Test Finished. Check output in: {output_normalized_dir.resolve()} ---")