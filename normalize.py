import json
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count # For multiprocessing

NUM_PROCESSES: int | None = 6 
FFMPEG_TIMEOUT_SECONDS = 600

TARGET_LUFS = -20.0    # Integrated loudness target in LUFS (moderate for speech)
TARGET_TP = -1.5        # True-peak target in dBTP
SAMPLE_RATE = 44100

def run_ffmpeg_command(cmd: list) -> subprocess.CompletedProcess:
    """
    Run an ffmpeg command via subprocess and return the CompletedProcess.
    Raises CalledProcessError on failure or TimeoutError on timeout.
    """
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False, 
            timeout=FFMPEG_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as e:
        error_message = f"ffmpeg command timed out after {FFMPEG_TIMEOUT_SECONDS}s.\n"
        error_message += f"Command: {' '.join(cmd)}\n"
        stdout_str = e.stdout.decode(errors='replace') if isinstance(e.stdout, bytes) else str(e.stdout or "")
        stderr_str = e.stderr.decode(errors='replace') if isinstance(e.stderr, bytes) else str(e.stderr or "")
        if stdout_str: error_message += f"Stdout: {stdout_str.strip()}\n"
        if stderr_str: error_message += f"Stderr: {stderr_str.strip()}\n"
        raise TimeoutError(error_message.strip()) from e

    if result.returncode != 0:
        result.check_returncode()
    return result

def measure_loudness(input_path: Path) -> dict:
    """
    First pass: measure loudness and dynamic range with ffmpeg loudnorm filter, return JSON stats.
    """
    filter_str = f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:print_format=json"
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(input_path),
        "-af", filter_str,
        "-f", "null", "-"
    ]
    print(f"[{input_path.name}] Measuring loudness...")
    completed = run_ffmpeg_command(cmd)

    stderr = completed.stderr
    start = stderr.rfind('{')
    if start == -1:
        raise ValueError(f"Could not find JSON start '{{' in ffmpeg output for {input_path.name}:\nStderr: {stderr}")
    
    end = stderr.find('}', start) + 1
    if end == 0:
        raise ValueError(f"Could not find JSON end '}}' in ffmpeg output for {input_path.name}:\nStderr: {stderr}")
        
    json_str = stderr[start:end]
    try:
        stats = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from ffmpeg output for {input_path.name}:\nError: {e}\nJSON string attempt: '{json_str}'\nFull stderr:\n{stderr}")
    return stats

def normalize_and_convert(input_path: Path, stats: dict, output_folder: Path) -> None:
    """
    Second pass: apply loudness normalization and convert to mono, preserving natural dynamics.
    Output as FLAC (lossless) to avoid any quality degradation.
    """
    measured_I      = stats['input_i']
    measured_LRA    = stats['input_lra']
    measured_TP     = stats['input_tp']
    measured_thresh = stats['input_thresh']
    offset          = stats.get('target_offset', 0.0)

    loudnorm_filter = (
        f"loudnorm=I={TARGET_LUFS}:LRA={measured_LRA}:TP={TARGET_TP}:"
        f"measured_I={measured_I}:measured_LRA={measured_LRA}:measured_TP={measured_TP}:"
        f"measured_thresh={measured_thresh}:offset={offset}:linear=true:print_format=summary"
    )

    output_path = output_folder / input_path.with_suffix('.flac').name

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(input_path), # Added -y to overwrite
        "-af", loudnorm_filter,
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "flac",
        str(output_path)
    ]
    print(f"[{input_path.name}] Normalizing & converting to {output_path.name}...")
    run_ffmpeg_command(cmd)


def process_single_file(input_path: Path, output_folder: Path) -> tuple[Path, Exception | None]:
    """
    Worker function: processes a single FLAC file (measure loudness, then normalize).
    Returns a tuple (input_path, None) on success, or (input_path, Exception) on failure.
    """
    try:
        stats = measure_loudness(input_path)
        normalize_and_convert(input_path, stats, output_folder)
        print(f"[{input_path.name}] Successfully processed.")
        return input_path, None
    except subprocess.CalledProcessError as e:
        error_detail = f"FFMPEG command failed for '{input_path.name}'.\n"
        error_detail += f"Command: '{' '.join(e.cmd)}'. Return code: {e.returncode}.\n"
        if e.stderr: error_detail += f"Stderr: {e.stderr.strip()}\n"
        if e.stdout: error_detail += f"Stdout: {e.stdout.strip()}\n"
        print(f"[{input_path.name}] Error: {error_detail.splitlines()[0]}...")
        return input_path, Exception(error_detail.strip())
    except TimeoutError as e:
        print(f"[{input_path.name}] Error: {str(e).splitlines()[0]}...")
        return input_path, e
    except ValueError as e:
        print(f"[{input_path.name}] Data processing error: {type(e).__name__}: {str(e).splitlines()[0]}...")
        return input_path, e
    except Exception as e:
        print(f"[{input_path.name}] An unexpected error occurred: {type(e).__name__}: {str(e).splitlines()[0]}...")
        return input_path, e

def batch_process_single_directory(input_folder: Path, output_folder: Path, num_processes: int | None) -> tuple[int, list[tuple[Path, str]]]:
    """
    Process all FLAC files in a single input_folder using multiple processes.
    Returns (number_successful, list_of_failed_items), where failed_items are (path, error_string).
    """
    effective_num_processes = num_processes
    if effective_num_processes is None:
        effective_num_processes = cpu_count()
    effective_num_processes = max(1, effective_num_processes)

    print(f"Processing directory: {input_folder.resolve()} -> {output_folder.resolve()}")

    flac_files = sorted(list(input_folder.glob("*.flac")))
    if not flac_files:
        print(f"No .flac files found in {input_folder}")
        return 0, []

    output_folder.mkdir(parents=True, exist_ok=True)

    tasks = [(fl, output_folder) for fl in flac_files]
    
    actual_num_processes = min(effective_num_processes, len(flac_files))
    if actual_num_processes < effective_num_processes:
        print(f"Using {actual_num_processes} processes (capped by number of files: {len(flac_files)}).")
    else:
        print(f"Using {actual_num_processes} processes (requested: {effective_num_processes}).")

    successful_count = 0
    failed_items: list[tuple[Path, str]] = []

    with Pool(processes=actual_num_processes) as pool:
        results = pool.starmap(process_single_file, tasks)

    for file_path, error_obj in results:
        if error_obj is None:
            successful_count += 1
        else:
            failed_items.append((file_path, str(error_obj)))

    print(f"\nFinished processing for directory {input_folder.name}.")
    print(f"  Successfully processed: {successful_count} files.")
    if failed_items:
        print(f"  Failed to process {len(failed_items)} files:")
        for f_path, err_msg in failed_items:
            print(f"    - {f_path.name}: {err_msg.splitlines()[0]}")
    return successful_count, failed_items

def orchestrate_subfolder_processing(
    base_input_dir: Path, 
    base_output_dir: Path, 
    subfolder_names: list[str], 
    num_processes: int | None
):
    """
    Processes FLAC files in specified subdirectories under base_input_dir,
    saving results to corresponding subdirectories in base_output_dir.
    This function orchestrates calls to batch_process_single_directory for each subfolder.
    """
    total_successfully_processed = 0
    total_failed_count = 0
    all_failed_details: list[tuple[str, Path, str]] = [] 
    skipped_folders_count = 0

    print(f"Starting batch processing for specified subfolders...")
    print(f"Base Input Directory: {base_input_dir.resolve()}")
    print(f"Base Output Directory: {base_output_dir.resolve()}")
    print(f"Subfolders to process: {', '.join(subfolder_names)}")
    if num_processes:
        print(f"Requested number of processes: {num_processes} (FFMPEG timeout per file: {FFMPEG_TIMEOUT_SECONDS}s)")
    else:
        print(f"Requested number of processes: Auto (CPU cores) (FFMPEG timeout per file: {FFMPEG_TIMEOUT_SECONDS}s)")
    print("-" * 50)

    for subfolder_name in subfolder_names:
        current_input_folder = base_input_dir / subfolder_name / "vocals"
        current_output_folder = base_output_dir / subfolder_name / "vocals_normalized"

        print(f"\n--- Processing subfolder: {subfolder_name} ---")

        if not current_input_folder.is_dir():
            print(f"⚠️ WARNING: Input folder '{current_input_folder.resolve()}' does not exist. Skipping.")
            skipped_folders_count += 1
            print("-" * 30)
            continue

        successful_count, failed_items = batch_process_single_directory(
            current_input_folder, 
            current_output_folder, 
            num_processes
        )
        
        total_successfully_processed += successful_count
        total_failed_count += len(failed_items)
        for item_path, item_error_str in failed_items:
            all_failed_details.append((subfolder_name, item_path, item_error_str))
        
        print("-" * 30)

    print("\n--- Overall Batch Processing Summary ---")
    if skipped_folders_count > 0:
        print(f"Total subfolders skipped (not found): {skipped_folders_count}")
    print(f"Total files successfully processed across all subfolders: {total_successfully_processed}")
    print(f"Total files failed to process across all subfolders: {total_failed_count}")
    if all_failed_details:
        print("Details of failed files (first line of error shown, see logs above for full details):")
        for sf_name, f_path, err_str in all_failed_details:
            first_line_error = err_str.splitlines()[0] if err_str and err_str.strip() else "Unknown error"
            print(f"  - In '{sf_name}/{f_path.name}': {first_line_error}")
    print("Batch processing for all specified directories complete.")


if __name__ == '__main__':
    BASE_INPUT_DIR = Path("/home/taresh/Downloads/anime/audios")
    BASE_OUTPUT_DIR = Path("/home/taresh/Downloads/anime/audios")
    SUBFOLDER_NAMES_TO_PROCESS = ["Rezero_s1", "Rezero_s2", "Rezero_s3p1", "Rezero_s3p2"]

    orchestrate_subfolder_processing(
        base_input_dir=BASE_INPUT_DIR,
        base_output_dir=BASE_OUTPUT_DIR,
        subfolder_names=SUBFOLDER_NAMES_TO_PROCESS,
        num_processes=NUM_PROCESSES
    )
    
    print("All processing finished.")