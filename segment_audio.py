import json
import os
import soundfile as sf
import re
import math
from typing import List, Dict
from tqdm.auto import tqdm
import numpy as np # Import numpy

def _time_str_to_seconds(time_str: str) -> float:
    """Converts HH:MM:SS.mmm or MM:SS.mmm string to seconds."""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    elif len(parts) == 1:
        return float(parts[0])
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def _sanitize_filename(name: str) -> str:
    """Removes or replaces characters invalid for filenames/directory names."""
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    name = name.replace(' ', '_')
    return name

def segment_audio_by_speaker(
    json_file_path: str,
    audio_file_path: str,
    output_base_dir: str,
    target_speakers: List[str],
    padding_seconds: float = 1.0 # Define padding duration here
):
    """
    Segments an audio file by speaker, saving .wav clips into per-speaker folders.
    Adds a fixed padding to the end of each segment.
    """
    print(f"Starting audio segmentation with {padding_seconds:.2f}s end padding...")

    # Validate inputs
    if not os.path.isfile(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Load JSON
    print(f"Loading JSON from: {json_file_path}")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            segments_data = json.load(f)
        if not isinstance(segments_data, list):
            raise ValueError("JSON must be a list of segment objects.")
    except Exception as e:
        print(f"Error loading or parsing JSON: {e}")
        raise

    # Load audio metadata
    print(f"Loading audio metadata from: {audio_file_path}")
    try:
        info = sf.info(audio_file_path)
        samplerate = info.samplerate
        subtype = info.subtype
        print(f"Audio info: Rate={samplerate}, Subtype={subtype}, Channels={info.channels}")
    except Exception as e:
        print(f"Error reading audio metadata: {e}")
        raise

    # Read full audio
    print("Loading full audio file...")
    try:
        full_audio, sr_read = sf.read(audio_file_path, dtype='float64', always_2d=True)
        if sr_read != samplerate:
            print(f"Warning: read samplerate {sr_read} differs from info.samplerate {samplerate}. Using {samplerate}.")
        print(f"Audio loaded successfully (shape: {full_audio.shape}).")
    except Exception as e:
        print(f"Error reading full audio data: {e}")
        raise

    # Prepare output base
    os.makedirs(output_base_dir, exist_ok=True)

    # Counters
    counters: Dict[str, int] = {spkr: 0 for spkr in target_speakers}
    total_saved = 0
    skipped_count = 0

    for idx, seg in enumerate(tqdm(segments_data, desc="Processing Segments")):
        spkr = seg.get('speaker')
        start_str = seg.get('start')
        end_str = seg.get('end')
        transcript = seg.get('transcript', '') # Get transcript for checks, default to empty

        # Skip if not a target speaker or essential info is missing
        # Also skip 'SOUND' or 'UNKNOWN' tags if they somehow match target_speakers list
        if (spkr not in target_speakers or
                not start_str or not end_str or
                spkr in ["SOUND", "UNKNOWN"]):
            continue
            
        # Skip if transcript is null or empty (as these might be incorrectly labelled noise)
        if not transcript:
             # print(f"Skipping segment {idx} for speaker '{spkr}': Missing transcript.")
             skipped_count += 1
             continue


        try:
            # Time to samples
            t0 = _time_str_to_seconds(start_str)
            t1 = _time_str_to_seconds(end_str)

            # --- Add Padding ---
            t1_padded = t1 + padding_seconds
            # -----------------

            if t1_padded <= t0:
                print(f"Warning: Skipping segment {idx} for '{spkr}': Padded end time ({t1_padded:.3f}s) is not after start time ({t0:.3f}s).")
                skipped_count += 1
                continue

            # Calculate sample indices (using original start, padded end)
            i0 = max(0, math.floor(t0 * samplerate))
            # Use the padded time for the end sample calculation
            i1_padded = min(full_audio.shape[0], math.ceil(t1_padded * samplerate))

            if i1_padded <= i0:
                # This might happen if the original segment was tiny and padding didn't help
                # print(f"Skipping segment {idx} for '{spkr}': Sample range invalid after padding ({i0} >= {i1_padded}).")
                skipped_count += 1
                continue

            # Extract the padded audio chunk
            chunk = full_audio[i0:i1_padded]

            # Paths
            safe_spkr = _sanitize_filename(spkr)
            counters[spkr] += 1
            count = counters[spkr]
            base_filename = f"{safe_spkr}_{count:04d}" # Using speaker name + counter

            speaker_dir = os.path.join(output_base_dir, safe_spkr)
            wav_dir = os.path.join(speaker_dir, 'wav') # Create subfolder for wav
            txt_dir = os.path.join(speaker_dir, 'txt') # Create subfolder for txt

            os.makedirs(wav_dir, exist_ok=True)
            os.makedirs(txt_dir, exist_ok=True)

            wav_path = os.path.join(wav_dir, f"{base_filename}.wav")
            txt_path = os.path.join(txt_dir, f"{base_filename}.txt")


            # Write audio clip
            sf.write(wav_path, chunk, samplerate, subtype=subtype)

            # Write corresponding transcript
            with open(txt_path, 'w', encoding='utf-8') as f_txt:
                f_txt.write(transcript)

            total_saved += 1

        except ValueError as e:
             print(f"Warning: Skipping segment {idx} due to value error: {e}")
             skipped_count += 1
        except Exception as e:
             print(f"Warning: Unexpected error processing segment {idx} for '{spkr}': {e}")
             skipped_count += 1
             # import traceback # Uncomment for detailed debugging
             # traceback.print_exc() # Uncomment for detailed debugging


    print(f"\nSegmentation finished.")
    print(f"Total segments saved: {total_saved}")
    if skipped_count > 0:
         print(f"Total segments skipped (invalid time/data/no transcript): {skipped_count}")
    print("Segments saved per speaker:")
    for speaker, count in counters.items():
        if count > 0:
            print(f"  - {speaker}: {count}")

if __name__ == '__main__':
    # --- Configuration ---
    JSON_FILE = 'temp/output5.json' # Path to your Gemini JSON output
    AUDIO_FILE = 'temp/vocals/aot_e10_vocals.wav' # Path to the original audio file Gemini processed
    OUTPUT_DIR = 'temp/segmented_vocals_e10_padded' # Base directory for segmented files
    SPEAKERS = ['Armin Arlert', 'Eren Yeager', 'Mikasa Ackerman'] # List of speaker labels you want
    PADDING = 1.0 # Seconds of padding to add to the end

    # --- Run ---
    try:
        segment_audio_by_speaker(
            json_file_path=JSON_FILE,
            audio_file_path=AUDIO_FILE,
            output_base_dir=OUTPUT_DIR,
            target_speakers=SPEAKERS,
            padding_seconds=PADDING
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")