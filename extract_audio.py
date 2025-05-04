import subprocess
import json
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List

# === Configuration ===
INPUT_DIR = Path("/home/taresh/Downloads/anime/videos/Konosuba_s2")
OUTPUT_DIR = Path("/home/taresh/Downloads/anime/audios/Konosuba_s2/orig")
PREFERRED_LANG_CODES: List[str] = ["eng", "en", "english"]  
OUTPUT_SAMPLE_RATE: Optional[str] = "44100"   # e.g. "48000" to resample, or None to keep original
OUTPUT_CODEC = "flac"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def probe_audio_streams(input_path: Path) -> List[dict]:
    """
    Run ffprobe to list audio streams. Returns list of stream dicts.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-select_streams", "a",
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    result.check_returncode()
    info = json.loads(result.stdout)
    return info.get("streams", [])


def select_audio_stream(streams: List[dict], preferred: List[str]) -> Optional[int]:
    """
    Choose stream index matching one of preferred language codes, or fallback to first.
    """
    # Try to match tag language
    for code in preferred:
        for s in streams:
            lang = s.get("tags", {}).get("language", "").lower()
            if lang == code.lower():
                return s.get("index")
    # Fallback: first audio stream
    if streams:
        return streams[0].get("index")
    return None


def extract_audio(input_path: Path, output_path: Path) -> bool:
    """
    Extracts the chosen audio stream to a FLAC file, preserving sample rate unless overridden.
    """
    streams = probe_audio_streams(input_path)
    idx = select_audio_stream(streams, PREFERRED_LANG_CODES)
    if idx is None:
        print(f"❌ No audio streams found in {input_path.name}")
        return False
    spec = f"0:{idx}"
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-map", spec,
        "-c:a", OUTPUT_CODEC
    ]
    if OUTPUT_SAMPLE_RATE:
        cmd += ["-ar", OUTPUT_SAMPLE_RATE]
    cmd.append(str(output_path))

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Extracted {input_path.name} → {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore")
        print(f"❌ Failed {input_path.name}: {err.splitlines()[-1]}")
        return False


def batch_extract():
    """
    Recursively scan INPUT_DIR for video files and extract audio as FLAC.
    """
    video_exts = {'.mkv', '.mp4', '.avi', '.mov'}
    files = list(INPUT_DIR.rglob("*"))
    for f in tqdm(files, desc="Extracting audio", unit="file"):
        if f.is_file() and f.suffix.lower() in video_exts:
            relative_path = f.relative_to(INPUT_DIR)
            output_subdir = OUTPUT_DIR / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True) 
            out = output_subdir / f.with_suffix('.flac').name
            extract_audio(f, out)


if __name__ == '__main__':
    batch_extract()
