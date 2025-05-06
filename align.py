import warnings
warnings.filterwarnings("ignore")

import os
import glob
import torch
import soundfile as sf
import numpy as np
import math
import re
import json
import tempfile
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCTC, AutoTokenizer

from ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,
                                    postprocess_results, preprocess_text, load_audio)

# --- Custom Model Loader ---
def load_alignment_model_custom(device: str, model_path: str, dtype: torch.dtype = torch.float32):
    """Loads alignment model and tokenizer without unsupported args."""
    print(f"Loading alignment model: {model_path}...")
    try:
        model = AutoModelForCTC.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Alignment model loaded.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading alignment model '{model_path}': {e}")
        raise

# --- Helper Functions ---
def _time_str_to_seconds(time_str: str) -> float:
    """Converts MM:SS.mmm string to seconds."""
    parts = time_str.split(':')
    try:
        if len(parts) == 2:
            m, s = map(float, parts)
            if s >= 60.0: m += math.floor(s / 60.0); s %= 60.0
            return m * 60 + s
        elif len(parts) == 1: return float(parts[0])
        else: raise ValueError(f"Expected MM:SS.mmm format, got {len(parts)} parts.")
    except ValueError as e:
         raise ValueError(f"Invalid time format: '{time_str}'. Error: {e}")

def _sanitize_filename(name: str) -> str:
    """Creates a safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(' ', '_')
    return name
# --- End Helper Functions ---

def align_single_temp_file(
    temp_audio_path: str, transcript: str, alignment_model, alignment_tokenizer,
    device: str, language: str = 'en', batch_size: int = 8
) -> Optional[Tuple[float, float]]:
    """Aligns a temporary audio file, returns relative start/end times of content."""
    # (Function implementation remains the same as the previous correct version)
    if not transcript: return None
    try:
        model_dtype = alignment_model.dtype
        audio_waveform = load_audio(temp_audio_path, dtype=model_dtype, device=device)
        if audio_waveform is None or audio_waveform.nelement() == 0: return None

        with torch.no_grad():
             emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=batch_size)

        tokens_starred, text_starred = preprocess_text(
            transcript, romanize=True, language=language, split_size='word', star_frequency='edges'
        )
        segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        first_word, last_word = None, None
        for seg in word_timestamps:
            if seg.get('text') and seg['text'] != '<star>':
                if first_word is None: first_word = seg
                last_word = seg

        if first_word and last_word and 'start' in first_word and 'end' in last_word:
            start_rel, end_rel = first_word['start'], last_word['end']
            return (start_rel, end_rel) if end_rel > start_rel else None
        else:
            print(f"  Warning: Could not extract valid start/end from alignment for {os.path.basename(temp_audio_path)}.")
            return None
    except Exception as e:
        print(f"  Error during alignment for {os.path.basename(temp_audio_path)}: {e}")
        return None


def process_episodes(
    input_json_dir: str,
    original_audio_dir: str,
    output_base_dir: str,
    target_speakers: List[str],
    aligner_model_name: str,
    language: str = 'en',
    padding_seconds: float = 0.75,
    use_gpu: bool = True,
    min_duration_seconds: float = 0.4, # New parameter for min duration
    min_words_in_transcript: int = 2   # New parameter for min words
    ):
    """Processes JSONs, aligns padded segments, filters short/simple ones, saves refined clips."""

    print("Starting Precise Segmentation Workflow...")
    # ... (Initial prints remain the same) ...
    print(f"Min Duration Filter : {min_duration_seconds}s")
    print(f"Min Words Filter    : {min_words_in_transcript}")


    # --- Setup Device & Load Model (Once) ---
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    try:
        compute_dtype = torch.float16 if device == "cuda" else torch.float32
        alignment_model, alignment_tokenizer = load_alignment_model_custom(
            device, aligner_model_name, compute_dtype
        )
    except Exception as e:
        print(f"FATAL: Could not load alignment model. Exiting. Error: {e}")
        return

    # --- Iterate Episodes (JSON files) ---
    json_files = glob.glob(os.path.join(input_json_dir, '*.json'))
    if not json_files:
        print(f"Error: No .json files found in '{input_json_dir}'.")
        return
    print(f"Found {len(json_files)} JSON files to process.")

    total_processed_segments = 0
    total_skipped_segments = 0

    for json_file_path in tqdm(json_files, desc="Processing Episodes"):
        episode_name = Path(json_file_path).stem
        print(f"\n--- Episode: {episode_name} ---")
        original_audio_path = Path(original_audio_dir) / f"{episode_name}.flac"

        if not original_audio_path.is_file():
            print(f"Error: Original audio not found at '{original_audio_path}'. Skipping episode.")
            total_skipped_segments += 1
            continue

        # --- Load Full Audio & JSON ---
        try:
            print(f"  Loading audio: {original_audio_path.name}...")
            audio_info = sf.info(str(original_audio_path))
            original_samplerate = audio_info.samplerate
            original_subtype = audio_info.subtype
            full_audio_data, _ = sf.read(str(original_audio_path), dtype='float64', always_2d=True)

            with open(json_file_path, 'r', encoding='utf-8') as f:
                segments_data = json.load(f)
            if not isinstance(segments_data, list): raise ValueError("JSON not a list")
            print(f"  Loaded {len(segments_data)} segments from JSON.")
        except Exception as e:
            print(f"Error loading audio or JSON for {episode_name}: {e}. Skipping episode.")
            total_skipped_segments += len(segments_data) if isinstance(segments_data, list) else 1
            continue

        # --- Process Segments ---
        segment_counters: Dict[str, int] = {spkr: 0 for spkr in target_speakers}
        ep_processed = 0
        ep_skipped = 0
        ep_filtered_duration = 0
        ep_filtered_transcript = 0

        for index, segment in enumerate(tqdm(segments_data, desc=f"Segments", leave=False)):
            speaker = segment.get('speaker')
            transcript = segment.get('transcript')
            start_str = segment.get('start')
            end_str = segment.get('end')

            # --- Filter by Target Speaker ---
            if speaker not in target_speakers:
                continue # Silently skip non-target speakers

            # --- Basic Data Validation ---
            if not transcript or not start_str or not end_str:
                ep_skipped += 1
                continue

            temp_wav_path = None
            temp_txt_path = None
            try:
                gemini_start_time = _time_str_to_seconds(start_str)
                gemini_end_time = _time_str_to_seconds(end_str)
                if gemini_end_time <= gemini_start_time:
                     ep_skipped += 1
                     continue

                # --- *** FILTERING LOGIC *** ---
                # 1. Duration Filter
                duration = gemini_end_time - gemini_start_time
                if duration < min_duration_seconds:
                    # print(f"  Filter: Skipping segment {index} (Speaker: {speaker}): Duration {duration:.3f}s < {min_duration_seconds}s")
                    ep_filtered_duration += 1
                    continue

                # 2. Transcript Content Filter
                words = transcript.split()
                word_count = len(words)
                is_bracketed_expression = transcript.startswith(('(', '[')) and transcript.endswith((')', ']'))

                # Filter if word count is too low OR if it looks like a bracketed sound effect/note
                if word_count < min_words_in_transcript or is_bracketed_expression:
                    # print(f"  Filter: Skipping segment {index} (Speaker: {speaker}): Word count {word_count} < {min_words_in_transcript} or bracketed: '{transcript[:50]}...'")
                    ep_filtered_transcript += 1
                    continue
                # --- *** END FILTERING LOGIC *** ---


                # --- Pad and Extract Temporary Audio ---
                padded_end_time = gemini_end_time + padding_seconds
                start_sample = max(0, math.floor(gemini_start_time * original_samplerate))
                padded_end_sample = min(full_audio_data.shape[0], math.ceil(padded_end_time * original_samplerate))
                if start_sample >= padded_end_sample: continue

                padded_audio_chunk = full_audio_data[start_sample:padded_end_sample]

                # --- Create & Run Aligner on Temporary Files ---
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_f, \
                     tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False, encoding='utf-8') as tmp_txt_f:
                    temp_wav_path = tmp_wav_f.name
                    temp_txt_path = tmp_txt_f.name
                    sf.write(temp_wav_path, padded_audio_chunk, original_samplerate, subtype='PCM_16')
                    tmp_txt_f.write(transcript)

                alignment_times = align_single_temp_file(
                    temp_wav_path, transcript, alignment_model, alignment_tokenizer, device, language
                )

                if alignment_times is None:
                    # print(f"Warning: Alignment failed for segment {index} (Speaker: {speaker}). Skipping.")
                    ep_skipped += 1
                    continue

                relative_start_time, relative_end_time = alignment_times
                final_start_abs = gemini_start_time # Use Gemini start
                final_end_abs = gemini_start_time + relative_end_time # Use aligned end

                if final_end_abs <= final_start_abs: continue

                # --- Extract Final Audio Clip ---
                final_start_sample = max(0, math.floor(final_start_abs * original_samplerate))
                final_end_sample = min(full_audio_data.shape[0], math.ceil(final_end_abs * original_samplerate))
                if final_start_sample >= final_end_sample: continue

                final_audio_chunk = full_audio_data[final_start_sample:final_end_sample]

                # --- Prepare Output Paths and Save ---
                sanitized_speaker = _sanitize_filename(speaker)
                segment_counters[speaker] += 1
                file_counter = segment_counters[speaker]
                base_filename = f"{sanitized_speaker}_{file_counter:04d}"

                speaker_output_dir = Path(output_base_dir) / episode_name / sanitized_speaker
                speaker_output_dir.mkdir(parents=True, exist_ok=True)

                final_flac_path = speaker_output_dir / f"{base_filename}.flac"
                final_txt_path = speaker_output_dir / f"{base_filename}.txt"

                sf.write(str(final_flac_path), final_audio_chunk, original_samplerate, subtype=original_subtype)
                with open(final_txt_path, 'w', encoding='utf-8') as f_txt:
                    f_txt.write(transcript)

                ep_processed += 1

            except (ValueError, KeyError) as e:
                 # print(f"Warning: Skipping segment {index} (Speaker: {speaker}) due to data error: {e}")
                 ep_skipped += 1
            except Exception as e_inner:
                 print(f"Error processing segment {index} (Speaker: {speaker}): {e_inner}")
                 ep_skipped += 1
            finally:
                 # Cleanup Temporary Files
                 for tmp_path in [temp_wav_path, temp_txt_path]:
                     if tmp_path and os.path.exists(tmp_path):
                          try: os.remove(tmp_path)
                          except Exception: pass

        print(f"  Finished episode {episode_name}. Processed: {ep_processed}, Skipped (Errors/Missing): {ep_skipped}, Filtered (Duration): {ep_filtered_duration}, Filtered (Transcript): {ep_filtered_transcript}")
        total_processed_segments += ep_processed
        # Accumulate total skipped/filtered counts
        total_skipped_segments += ep_skipped + ep_filtered_duration + ep_filtered_transcript


    print("-" * 30)
    print("\nBatch processing complete.")
    print(f"Total segments processed and saved: {total_processed_segments}")
    print(f"Total segments skipped/filtered : {total_skipped_segments}")


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_JSON_DIR = "/home/taresh/Downloads/anime/audios/Dandadan/gemini_outputs5_1ch_t0.5"
    ORIGINAL_AUDIO_DIR = "/home/taresh/Downloads/anime/audios/Dandadan/vocals_normalized"
    OUTPUT_BASE_DIR = "/home/taresh/Downloads/anime/audios/Dandadan/final_aligned_clips_filtered" # New output dir

    TARGET_SPEAKERS = ["Momo Ayase"]

    PADDING_SECONDS = 1
    MIN_DURATION_SECONDS = 0.4 # Filter segments shorter than 0.5 seconds
    MIN_WORDS_IN_TRANSCRIPT = 1 # Filter segments with 0 or 1 word

    ALIGNER_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    LANGUAGE = 'en'
    USE_GPU = True

    # --- Basic Path Checks ---
    if not os.path.isdir(INPUT_JSON_DIR) or not os.path.isdir(ORIGINAL_AUDIO_DIR):
        print(f"ERROR: Input JSON directory ('{INPUT_JSON_DIR}') or Original Audio directory ('{ORIGINAL_AUDIO_DIR}') not found.")
    elif not TARGET_SPEAKERS:
        print(f"ERROR: TARGET_SPEAKERS list cannot be empty.")
    else:
        process_episodes(
            input_json_dir=INPUT_JSON_DIR,
            original_audio_dir=ORIGINAL_AUDIO_DIR,
            output_base_dir=OUTPUT_BASE_DIR,
            target_speakers=TARGET_SPEAKERS,
            aligner_model_name=ALIGNER_MODEL,
            language=LANGUAGE,
            padding_seconds=PADDING_SECONDS,
            use_gpu=USE_GPU,
            min_duration_seconds=MIN_DURATION_SECONDS, # Pass new args
            min_words_in_transcript=MIN_WORDS_IN_TRANSCRIPT # Pass new args
        )