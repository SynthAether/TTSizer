import subprocess
import json
from tqdm import tqdm
from pathlib import Path
import tempfile
from typing import Optional, List
import multiprocessing # Added for multiprocessing

# === Main Configuration ===
BASE_INPUT_DIR_MAIN = Path("/home/taresh/Downloads/anime/videos")
BASE_OUTPUT_DIR_MAIN = Path("/home/taresh/Downloads/anime/audios")

# List of folder names within BASE_INPUT_DIR_MAIN to process
FOLDERS_TO_PROCESS: List[str] = [
    # "fate_1",
    # "fate_2",
    "Madoka_Magica",
    # "SAO_1", # Corrected missing comma
    # "konosuba"
    # "bunny_girl"
]

PREFERRED_LANG_CODES: List[str] = ["eng", "en", "english"]
MAX_WORKERS: Optional[int] = 6
OUTPUT_SAMPLE_RATE = "44100"
FINAL_OUTPUT_CODEC = "flac"
RESOLUTION_THRESHOLD_FOR_AAC_STEP = 480
INTERMEDIATE_AAC_BITRATE = "40k"


def probe_audio_streams(input_path: Path) -> List[dict]:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-select_streams", "a",
        str(input_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return info.get("streams", [])
    except subprocess.CalledProcessError as e:
        tqdm.write(f"❌ ffprobe (audio) failed for {input_path.name}: {e.stderr.decode(errors='ignore')}")
        return []
    except json.JSONDecodeError:
        tqdm.write(f"❌ ffprobe (audio) returned invalid JSON for {input_path.name}")
        return []


def probe_video_height(input_path: Path) -> Optional[int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-select_streams", "v:0",
        str(input_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        streams = info.get("streams", [])
        if streams and isinstance(streams, list) and len(streams) > 0:
            if "height" in streams[0]:
                return int(streams[0]["height"])
        return None
    except subprocess.CalledProcessError:
        return None
    except (json.JSONDecodeError, ValueError, TypeError):
        tqdm.write(f"❌ ffprobe (video) returned unexpected data for {input_path.name}")
        return None


def select_audio_stream(streams: List[dict], preferred: List[str]) -> Optional[int]:
    for code in preferred:
        for s in streams:
            lang_tags = s.get("tags", {})
            if lang_tags:
                lang = lang_tags.get("language", "").lower()
                if lang == code.lower():
                    return s.get("index")
    if streams:
        return streams[0].get("index")
    return None


def extract_and_convert_audio(input_path: Path, output_path: Path) -> bool:
    """
    Worker function to extract audio from a single video file.
    Global constants like PREFERRED_LANG_CODES, RESOLUTION_THRESHOLD_FOR_AAC_STEP, etc.,
    are accessible by worker processes.
    """
    audio_streams = probe_audio_streams(input_path)
    audio_stream_idx = select_audio_stream(audio_streams, PREFERRED_LANG_CODES)

    if audio_stream_idx is None:
        tqdm.write(f"❌ No suitable audio streams found in {input_path.name}")
        return False

    audio_map_spec = f"0:{audio_stream_idx}"
    video_height = probe_video_height(input_path)

    perform_aac_step = False
    if video_height is not None and video_height > RESOLUTION_THRESHOLD_FOR_AAC_STEP:
        perform_aac_step = True
        tqdm.write(f"ℹ️ Video {input_path.name} ({video_height}p) > {RESOLUTION_THRESHOLD_FOR_AAC_STEP}p. Applying intermediate AAC step.")
    elif video_height is not None:
         tqdm.write(f"ℹ️ Video {input_path.name} ({video_height}p) <= {RESOLUTION_THRESHOLD_FOR_AAC_STEP}p. Direct FLAC conversion.")
    else:
        tqdm.write(f"ℹ️ Video height for {input_path.name} unknown. Assuming direct FLAC conversion.")

    if perform_aac_step:
        with tempfile.TemporaryDirectory(prefix="audio_conv_") as temp_dir_name:
            temp_aac_path = Path(temp_dir_name) / f"{input_path.stem}_temp.aac"
            cmd_to_aac = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-map", audio_map_spec,
                "-c:a", "aac",
                "-b:a", INTERMEDIATE_AAC_BITRATE,
                "-ar", OUTPUT_SAMPLE_RATE,
                str(temp_aac_path)
            ]
            try:
                subprocess.run(cmd_to_aac, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                err = e.stderr.decode(errors="ignore")
                tqdm.write(f"❌ Stage 1 (AAC for >{RESOLUTION_THRESHOLD_FOR_AAC_STEP}p) Failed for {input_path.name}: {err.splitlines()[-1] if err.splitlines() else 'Unknown FFmpeg error'}")
                return False

            cmd_to_flac = [
                "ffmpeg", "-y", "-i", str(temp_aac_path),
                "-c:a", FINAL_OUTPUT_CODEC,
                str(output_path)
            ]
            try:
                subprocess.run(cmd_to_flac, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                tqdm.write(f"✅ Converted {input_path.name} → {output_path.name} (via {INTERMEDIATE_AAC_BITRATE} AAC @ {OUTPUT_SAMPLE_RATE}Hz)")
                return True
            except subprocess.CalledProcessError as e:
                err = e.stderr.decode(errors="ignore")
                tqdm.write(f"❌ Stage 2 (FLAC from AAC) Failed for {input_path.name}: {err.splitlines()[-1] if err.splitlines() else 'Unknown FFmpeg error'}")
                return False
    else:
        cmd_direct_to_flac = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-map", audio_map_spec,
            "-c:a", FINAL_OUTPUT_CODEC,
            "-ar", OUTPUT_SAMPLE_RATE,
            str(output_path)
        ]
        try:
            subprocess.run(cmd_direct_to_flac, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tqdm.write(f"✅ Directly converted {input_path.name} → {output_path.name} (FLAC @ {OUTPUT_SAMPLE_RATE}Hz)")
            return True
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="ignore")
            tqdm.write(f"❌ Direct FLAC conversion Failed for {input_path.name}: {err.splitlines()[-1] if err.splitlines() else 'Unknown FFmpeg error'}")
            return False
    return False


def process_single_directory(current_input_dir: Path, current_output_dir_for_orig: Path):
    """
    Scans current_input_dir for video files and extracts/converts audio in parallel.
    """
    current_output_dir_for_orig.mkdir(parents=True, exist_ok=True)
    # These prints happen once per directory, outside the parallel part
    print(f"\nProcessing directory: {current_input_dir}")
    print(f"Outputting 'orig' audio to: {current_output_dir_for_orig}")
    print(f"Using up to {MAX_WORKERS if MAX_WORKERS is not None else multiprocessing.cpu_count()} worker processes for this directory.")


    video_exts = {'.mkv', '.mp4', '.avi', '.mov', '.webm'}
    
    # 1. Collect all tasks (input_path, output_path pairs)
    tasks = []
    for f_path in current_input_dir.rglob("*"):
        if f_path.is_file() and f_path.suffix.lower() in video_exts:
            relative_path_within_current_input = f_path.relative_to(current_input_dir)
            output_target_subdir = current_output_dir_for_orig / relative_path_within_current_input.parent
            output_target_subdir.mkdir(parents=True, exist_ok=True)
            
            out_filename = f_path.with_suffix(f'.{FINAL_OUTPUT_CODEC.lower()}').name
            out_path = output_target_subdir / out_filename
            tasks.append((f_path, out_path))

    if not tasks:
        print(f"No video files found in {current_input_dir} to process.")
        return
    
    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
        results = list(tqdm(pool.starmap(extract_and_convert_audio, tasks),
                            total=len(tasks),
                            desc=f"Audio from {current_input_dir.name}",
                            unit="file",
                            leave=False)) 
    


def process_multiple_directories():
    """
    Iterates through FOLDERS_TO_PROCESS and calls processing for each.
    """
    if not FOLDERS_TO_PROCESS:
        print("No folders specified in FOLDERS_TO_PROCESS. Exiting.")
        return

    print(f"Starting batch audio extraction for {len(FOLDERS_TO_PROCESS)} specified folder(s)...")
    print(f"Base Input Directory: {BASE_INPUT_DIR_MAIN}")
    print(f"Base Output Directory: {BASE_OUTPUT_DIR_MAIN}")
    actual_max_workers = MAX_WORKERS if MAX_WORKERS is not None else multiprocessing.cpu_count()
    print(f"Configured to use up to {actual_max_workers} worker processes per directory.")


    for folder_name in tqdm(FOLDERS_TO_PROCESS, desc="Overall Progress (Folders)", unit="folder"):
        current_input_dir = BASE_INPUT_DIR_MAIN / folder_name
        current_output_dir_for_orig_audio = BASE_OUTPUT_DIR_MAIN / folder_name / "orig"

        if not current_input_dir.is_dir():
            print(f"\n⚠️  Warning: Input folder '{current_input_dir}' not found. Skipping.")
            continue
        
        process_single_directory(current_input_dir, current_output_dir_for_orig_audio)
    
    print("\nAll specified folders processed.")


if __name__ == '__main__':
    multiprocessing.freeze_support() # Good practice, especially for frozen executables
    process_multiple_directories()