import json
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Optional, List, Tuple
import yaml
from tqdm import tqdm
from ttsizer.utils.logger import get_logger

logger = get_logger("vocals_normalizer")

class VocalsNormalizer:
    """Normalizes audio files to target loudness (LUFS) and true peak (TP) levels using FFmpeg.

    This class uses FFmpeg to analyze audio files for their current loudness
    and then applies normalization to meet specified LUFS and true peak targets.
    It supports multiprocessing for batch operations and can skip already processed files.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initializes the VocalsNormalizer with configuration settings.

        Args:
            config: A dictionary containing settings for audio normalization,
                    typically from the 'vocals_normalizer' section of config.yaml.
        """
        self.lufs = config["target_lufs"]
        self.tp = config["target_tp"]
        self.sr = config["sample_rate"]
        self.codec = config.get("output_codec", "flac")
        self.timeout = config["ffmpeg_timeout_seconds"]
        self.procs = config.get("num_processes")
        self.skip_exist = config.get("skip_if_output_exists", True)

    def _run_ffmpeg(self, cmd: List[str], fname: str) -> subprocess.CompletedProcess:
        """Executes an FFmpeg command and handles potential errors, including timeout.

        Args:
            cmd: A list of strings representing the FFmpeg command and its arguments.
            fname: The name of the file being processed (for logging purposes).

        Returns:
            A subprocess.CompletedProcess object upon successful execution.
        
        Raises:
            subprocess.CalledProcessError: If FFmpeg returns a non-zero exit code.
            subprocess.TimeoutExpired: If the FFmpeg command exceeds the configured timeout.
        """
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=self.timeout
        )
        
        if result.returncode != 0:
            error_message = f"FFmpeg command failed for {fname} with exit code {result.returncode}."
            stderr_output = result.stderr.strip()
            if stderr_output:
                error_message += f"\nFFmpeg stderr:\n{stderr_output}"
            logger.error(error_message)
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=result.stdout, stderr=result.stderr
            )
        return result

    def _get_loudness(self, path: Path) -> Dict[str, Any]:
        """Measures loudness statistics of an audio file using FFmpeg's loudnorm filter.

        Args:
            path: Path to the input audio file.

        Returns:
            A dictionary containing loudness statistics (e.g., input_i, input_tp).
        
        Raises:
            ValueError: If FFmpeg output does not contain valid JSON loudness data.
            subprocess.CalledProcessError: If the underlying FFmpeg command fails.
        """
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
            error_msg = f"Could not find valid JSON loudness data in ffmpeg output for {path.name}. Stderr: {stderr[:500]}..."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return json.loads(stderr[start:end])

    def _normalize(self, in_path: Path, stats: Dict[str, Any], out_path: Path):
        """Applies loudness normalization to an audio file using measured statistics.

        Args:
            in_path: Path to the input audio file.
            stats: Loudness statistics previously obtained from _get_loudness.
            out_path: Path to save the normalized audio file.
        """
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
            "-ac", "1", # Force mono output
            "-ar", str(self.sr),
            "-c:a", self.codec,
            str(out_path)
        ]
        self._run_ffmpeg(cmd, in_path.name)

    def _process_file(self, in_path: Path, out_path: Path) -> Tuple[Path, Optional[str]]:
        """Processes a single audio file: measures loudness then applies normalization.

        Skips processing if the output file already exists and skip_exist is True.

        Args:
            in_path: Path to the input audio file.
            out_path: Path for the normalized output audio file.

        Returns:
            A tuple containing the input Path and an error message string if an error
            occurred (or "skipped_exists" if skipped), None otherwise for success.
        """
        if self.skip_exist and out_path.exists():
            return in_path, "skipped_exists"
            
        try:
            stats = self._get_loudness(in_path)
            self._normalize(in_path, stats, out_path)
            return in_path, None
        except Exception as e:
            logger.error(f"Error processing file {in_path.name}: {e}", exc_info=True)
            return in_path, f"Normalization failed: {type(e).__name__}"

    def process_directory(self, in_dir: Path, out_dir: Path):
        """Normalizes all supported audio files in a directory structure using multiprocessing.
        
        Replicates the input directory structure in the output directory for normalized files.

        Args:
            in_dir: Root directory containing audio files to normalize.
            out_dir: Base directory to save normalized audio files.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Normalizing audio from: {in_dir.resolve()}")
        logger.info(f"Outputting normalized audio to: {out_dir.resolve()}")
        if self.skip_exist:
            logger.info("Skipping files if normalized version already exists.")

        exts = ["*.flac", "*.wav", "*.mp3", "*.aac", "*.m4a"]
        files = []
        for ext in exts:
            files.extend(list(in_dir.rglob(ext)))
        files = sorted(list(set(files)))

        if not files:
            logger.warning(f"No supported audio files found in {in_dir} or its subdirectories.")
            return

        tasks = []
        for in_path in files:
            rel_path = in_path.relative_to(in_dir)
            out_path = out_dir / rel_path.with_suffix(f".{self.codec}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((in_path, out_path))

        procs = self.procs if self.procs is not None else cpu_count()
        procs = max(1, min(procs, len(tasks)))

        logger.info(f"Found {len(tasks)} audio files to normalize. Using {procs} worker processes.")

        success_count = 0
        skipped_count = 0
        failed_files = []

        try:
            with Pool(processes=procs) as pool:
                results = []
                try:
                    for result_tuple in tqdm(pool.imap_unordered(self._process_file_wrapper, tasks), total=len(tasks), desc="Normalizing", unit="file"):
                        results.append(result_tuple)
                finally:
                    pool.close()
                    pool.join()
                
                for path, err_msg in results:
                    if err_msg is None:
                        success_count += 1
                    elif err_msg == "skipped_exists":
                        skipped_count += 1
                    else:
                        failed_files.append((path, err_msg))
        except Exception as e:
            logger.error(f"A critical error occurred during multiprocessing: {e}", exc_info=True)
            return

        logger.info(f"\nNormalization process finished for directory: {in_dir.name}")
        logger.info(f"Successfully normalized: {success_count} files")
        if skipped_count > 0:
            logger.info(f"Skipped (already exist): {skipped_count} files")
        if failed_files:
            logger.warning(f"Failed to normalize: {len(failed_files)} files. See earlier logs for details.")


    def _process_file_wrapper(self, args_tuple):
        """Wrapper for _process_file to enable its use with multiprocessing.Pool methods.

        This helper unpacks arguments for _process_file when called via pool.imap_unordered.

        Args:
            args_tuple: A tuple containing (in_path, out_path) for _process_file.

        Returns:
            The result of calling _process_file with the unpacked arguments.
        """
        return self._process_file(*args_tuple)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)

    project_cfg = cfg["project_setup"]
    vocals_extractor_cfg = cfg["vocals_extractor"]
    vocals_normalizer_cfg = cfg["vocals_normalizer"]

    base_output_dir = Path(project_cfg["output_base_dir"]) / project_cfg["series_name"]
    
    in_dir = base_output_dir / vocals_extractor_cfg["output_folder"]
    out_dir = base_output_dir / vocals_normalizer_cfg["output_folder"]
    
    print(f"Normalizing vocals for directory: {in_dir}")
    normalizer = VocalsNormalizer(vocals_normalizer_cfg)
    normalizer.process_directory(in_dir, out_dir)