import json
import subprocess
from pathlib import Path

INPUT_DIR = Path("/home/taresh/Downloads/anime/audios/Dandadan/vocals")       # Directory containing input .flac files
OUTPUT_DIR = Path("/home/taresh/Downloads/anime/audios/Dandadan/vocals_normalized")  # Where to save processed .flac files
TARGET_LUFS = -20.0    # Integrated loudness target in LUFS (moderate for speech)
TARGET_TP = -1.5        # True-peak target in dBTP

SAMPLE_RATE = 44100


def run_ffmpeg_command(cmd: list) -> subprocess.CompletedProcess:
    """
    Run an ffmpeg command via subprocess and return the CompletedProcess.
    Raises CalledProcessError on failure.
    """
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
    print(f"Measuring loudness for {input_path.name}...")
    completed = run_ffmpeg_command(cmd)

    stderr = completed.stderr
    start = stderr.find('{')
    end = stderr.rfind('}') + 1
    json_str = stderr[start:end]
    stats = json.loads(json_str)
    return stats


def normalize_and_convert(input_path: Path, stats: dict) -> None:
    """
    Second pass: apply loudness normalization and convert to mono, preserving natural dynamics.
    Output as FLAC (lossless) to avoid any quality degradation.
    """
    measured_I      = stats['input_i']
    measured_LRA    = stats['input_lra']  # preserve original dynamic range
    measured_TP     = stats['input_tp']
    measured_thresh = stats['input_thresh']
    offset          = stats.get('target_offset', 0.0)

    # Compose loudnorm filter:
    loudnorm_filter = (
        f"loudnorm=I={TARGET_LUFS}:LRA={measured_LRA}:TP={TARGET_TP}:"
        f"measured_I={measured_I}:measured_LRA={measured_LRA}:measured_TP={measured_TP}:"
        f"measured_thresh={measured_thresh}:offset={offset}:linear=true:print_format=summary"
    )

    # Prepare output path with .flac extension
    output_path = OUTPUT_DIR / input_path.with_suffix('.flac').name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(input_path),
        "-af", loudnorm_filter,
        "-ac", "1",        # convert to mono
        "-ar", str(SAMPLE_RATE),
        "-c:a", "flac",     # encode as FLAC (lossless)
        str(output_path)
    ]
    print(f"Normalizing & converting to mono as FLAC: {output_path.name}...")
    run_ffmpeg_command(cmd)


def batch_process():
    """
    Process all FLAC files in INPUT_DIR: measure loudness, then normalize & convert to mono FLAC without quality loss.
    """
    flac_files = list(INPUT_DIR.glob("*.flac"))
    if not flac_files:
        print(f"No .flac files found in {INPUT_DIR}")
        return

    for fl in flac_files:
        try:
            stats = measure_loudness(fl)
            normalize_and_convert(fl, stats)
        except Exception as e:
            print(f"Error processing {fl.name}: {e}")

    print("Batch processing complete.")


if __name__ == '__main__':
    batch_process()
