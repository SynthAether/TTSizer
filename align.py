import warnings
warnings.filterwarnings("ignore")

import os
import glob
import torch
import soundfile as sf
import numpy as np
import math
import re # Added import for re
from tqdm.auto import tqdm
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCTC, AutoTokenizer # Direct import for custom loader

# Assuming ctc_forced_aligner is installed or accessible in the environment
# We still need its functions other than load_alignment_model
try:
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        # load_alignment_model, # Don't import the original loader
        postprocess_results,
        preprocess_text,
        load_audio,
    )
except ImportError as e:
     print(f"Error: Could not import from 'ctc_forced_aligner'.")
     print(f"Please ensure the library is installed correctly: pip install ctc-forced-aligner")
     print(f"Original error: {e}")
     # Exit if the core library is missing
     import sys
     sys.exit(1)


# --- Custom Model Loader ---
def load_alignment_model_custom(
    device: str,
    model_path: str,
    dtype: torch.dtype = torch.float32,
):
    """
    Loads alignment model and tokenizer without attn_implementation argument.
    """
    print(f"Loading custom: Model={model_path}, Dtype={dtype}, Device={device}")
    try:
        model = (
            AutoModelForCTC.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path
            )
        print("Custom model/tokenizer loading successful.")
        return model, tokenizer
    except Exception as e:
        print(f"Error during custom model loading for '{model_path}': {e}")
        raise

# --- Helper Functions ---
def _time_str_to_seconds(time_str: str) -> float:
    """Converts HH:MM:SS.mmm or MM:SS.mmm string to seconds."""
    parts = time_str.split(':')
    try:
        if len(parts) == 3: # HH:MM:SS.mmm
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2: # MM:SS.mmm
            m, s = map(float, parts)
            return m * 60 + s
        elif len(parts) == 1: # S.mmm
            return float(parts[0])
        else:
            raise ValueError(f"Unexpected time format parts: {len(parts)}")
    except ValueError as e:
         raise ValueError(f"Invalid time format: '{time_str}'. Error: {e}")


def _sanitize_filename(name: str) -> str:
    """Removes or replaces characters invalid for filenames/directory names."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.replace(' ', '_')
    return name
# --- End Helper Functions ---


def align_single_file(
    audio_path: str,
    transcript: str,
    alignment_model, # Pass loaded model
    alignment_tokenizer, # Pass loaded tokenizer
    device: str,
    language: str = 'en',
    romanize: bool = False,
    batch_size: int = 8
) -> Optional[float]:
    """
    Performs forced alignment on a single audio file and transcript.
    Returns the precise end time (in seconds) of the last word in the alignment,
    or None if alignment fails or the transcript is empty.
    """
    # (Function implementation remains the same as the previous correct version)
    if not transcript:
        # print(f"  Skipping alignment for {os.path.basename(audio_path)}: Empty transcript.") # Reduced verbosity
        return None

    try:
        model_dtype = alignment_model.dtype
        audio_waveform = load_audio(audio_path, dtype=model_dtype, device=device)
        if audio_waveform is None or audio_waveform.nelement() == 0:
            print(f"  Skipping alignment for {os.path.basename(audio_path)}: Failed to load or empty audio.")
            return None

        with torch.no_grad():
             emissions, stride = generate_emissions(
                 alignment_model,
                 audio_waveform,
                 batch_size=batch_size,
             )

        tokens_starred, text_starred = preprocess_text(
            transcript,
            romanize=romanize,
            language=language,
            split_size='word',
            star_frequency='edges'
        )

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )

        spans = get_spans(tokens_starred, segments, blank_token)

        word_timestamps = postprocess_results(
            text_starred=text_starred,
            spans=spans,
            stride=stride,
            scores=scores
            )

        if word_timestamps:
            last_real_word_segment = None
            for seg in reversed(word_timestamps):
                if seg.get('text') and seg['text'] != '<star>':
                    last_real_word_segment = seg
                    break

            if last_real_word_segment and 'end' in last_real_word_segment:
                final_end_time = last_real_word_segment['end']
                return final_end_time
            else:
                 print(f"  Warning for {os.path.basename(audio_path)}: No non-star text segments found after alignment.")
                 return None
        else:
            print(f"  Warning for {os.path.basename(audio_path)}: Alignment produced no word timestamps.")
            return None
        
        # if word_timestamps:
        #     return word_timestamps[-1]['end']
        # return None 

    except FileNotFoundError:
        print(f"  Error: Audio file not found during alignment processing: {audio_path}")
        return None
    except Exception as e:
        print(f"  Error during alignment for {os.path.basename(audio_path)}: {type(e).__name__} - {e}")
        return None


def process_directory(
    input_audio_dir: str,
    input_txt_dir: str,
    output_audio_dir: str,
    aligner_model_name: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
    language: str = 'en',
    use_gpu: bool = True):
    """
    Processes a directory of audio segments and text files, performs forced
    alignment to find precise end times, and saves correctly clipped audio.
    """
    print("Starting precise audio segment clipping...")
    # ... (Initial prints and device setup remain the same) ...
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU (CUDA): {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        if use_gpu: print("CUDA not available, using CPU.")
        else: print("Using CPU.")

    print(f"Loading alignment model: {aligner_model_name}...")
    try:
        compute_dtype = torch.float16 if device == "cuda" else torch.float32
        alignment_model, alignment_tokenizer = load_alignment_model_custom(
            device=device, model_path=aligner_model_name, dtype=compute_dtype
        )
        print("Alignment model loaded successfully.")
    except Exception as e:
        print(f"Exiting due to model loading error.")
        return

    try:
        os.makedirs(output_audio_dir, exist_ok=True)
    except OSError as e:
         print(f"Error creating output directory {output_audio_dir}: {e}")
         return

    try:
        # Find WAV files recursively within the input audio directory
        # This assumes a structure like input_audio_dir/Speaker_Name/wav/audio.wav
        audio_files = glob.glob(os.path.join(input_audio_dir, '**', '*.wav'), recursive=True)

        # Filter out files that are not in a '/wav/' subdirectory if that's expected structure
        audio_files = [f for f in audio_files if os.path.basename(os.path.dirname(f)).lower() == 'wav']

        if not audio_files:
            print(f"Error: No .wav files found within a 'wav' subdirectory in '{input_audio_dir}'.")
            return
    except Exception as e:
         print(f"Error finding audio files in {input_audio_dir}: {e}")
         return

    print(f"Found {len(audio_files)} '.wav' files in 'wav' subdirectories to process.")
    processed_count = 0
    skipped_count = 0

    for audio_path in tqdm(audio_files, desc="Aligning and Clipping"):
        try:
            # --- Corrected Transcript Path Logic ---
            base_filename_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
            # Get the directory containing the audio file (e.g., .../Speaker_Name/wav)
            audio_file_dir = os.path.dirname(audio_path)
            # Get the parent directory (e.g., .../Speaker_Name)
            speaker_base_dir = os.path.dirname(audio_file_dir)
            # Get just the speaker name / final directory component
            speaker_subdir_name = os.path.basename(speaker_base_dir)

            # Construct the transcript path using the input_txt_dir, speaker subdir, 'txt', and filename
            transcript_path = os.path.join(input_txt_dir, speaker_subdir_name, 'txt', f"{base_filename_no_ext}.txt")
            # ----------------------------------------

            # --- Check for Transcript ---
            if not os.path.isfile(transcript_path):
                # Print the path it *actually* checked for debugging
                print(f"Warning: Transcript file not found for '{os.path.relpath(audio_path, input_audio_dir)}'. Expected at: '{transcript_path}'. Skipping.")
                skipped_count += 1
                continue

            # --- Read Transcript ---
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                if not transcript:
                     # print(f"Warning: Transcript file is empty for '{relative_path}'. Skipping.")
                     skipped_count += 1
                     continue
            except Exception as e:
                print(f"Warning: Error reading transcript '{transcript_path}': {e}. Skipping.")
                skipped_count += 1
                continue

            # --- Get Precise End Time using Aligner ---
            final_end_time = align_single_file(
                audio_path,
                transcript,
                alignment_model,
                alignment_tokenizer,
                device,
                language=language
            )

            if final_end_time is None:
                skipped_count += 1
                continue

            # --- Load Original Segment, Slice, and Save ---
            try:
                info = sf.info(audio_path)
                samplerate = info.samplerate
                subtype = info.subtype
                input_segment_audio, sr_read = sf.read(audio_path, dtype='float64', always_2d=True)

                start_sample = 0
                final_end_sample = min(input_segment_audio.shape[0], max(0, math.ceil(final_end_time * samplerate)))

                if final_end_sample <= start_sample:
                    print(f"Warning: Final sample range is invalid ({start_sample} >= {final_end_sample}) for '{os.path.relpath(audio_path, input_audio_dir)}' after alignment. Skipping.")
                    skipped_count += 1
                    continue

                output_audio_chunk = input_segment_audio[start_sample:final_end_sample]

                # Define output path, maintaining relative speaker structure but saving directly in output_audio_dir
                # Example: output_audio_dir / Speaker_Name_0001.wav
                # OR output_audio_dir / Speaker_Name / Speaker_Name_0001.wav
                # Let's save directly into output_audio_dir for simplicity here, adjust if needed
                output_filename = f"{base_filename_no_ext}.wav" # Use the original base filename
                output_path = os.path.join(output_audio_dir, output_filename)

                # If you want to replicate speaker subdirs in output:
                # output_speaker_dir = os.path.join(output_audio_dir, speaker_subdir_name)
                # os.makedirs(output_speaker_dir, exist_ok=True)
                # output_path = os.path.join(output_speaker_dir, output_filename)


                sf.write(output_path, output_audio_chunk, samplerate, subtype=subtype)
                processed_count += 1

            except FileNotFoundError:
                print(f"Error: Input audio file vanished during processing? '{audio_path}'. Skipping.")
                skipped_count +=1
            except Exception as e:
                print(f"Error processing/saving final clip for '{os.path.relpath(audio_path, input_audio_dir)}': {e}")
                skipped_count += 1

        except Exception as outer_e:
            print(f"An unexpected error occurred processing file '{audio_path}': {outer_e}")
            skipped_count += 1


    # ... (Final print summary remains the same) ...
    print(f"\nProcessing complete.")
    print(f"Successfully processed and saved: {processed_count} files.")
    print(f"Skipped files (missing transcript, empty audio/transcript, alignment errors, etc.): {skipped_count}")


if __name__ == "__main__":
    # --- Configuration ---
    # Directory with the input audio segments (SHOULD have Speaker/wav structure)
    INPUT_AUDIO_DIR = 'temp/segmented_vocals_e10_padded'
    # Directory with the corresponding transcript files (SHOULD have Speaker/txt structure)
    INPUT_TXT_DIR = 'temp/segmented_vocals_e10_padded'
    # Directory where the final clipped audio segments will be saved
    OUTPUT_CLIPPED_AUDIO_DIR = 'temp/segmented_vocals_e10_final2'

    # Aligner configuration
    ALIGNER_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner" # Or your preferred model
    LANGUAGE = 'en' # Set your language code
    USE_GPU = True  # Set to False to force CPU

    # --- Basic Path Checks ---
    paths_ok = True
    if not os.path.isdir(INPUT_AUDIO_DIR):
         print(f"ERROR: Input audio directory not found: {INPUT_AUDIO_DIR}")
         paths_ok = False
    if not os.path.isdir(INPUT_TXT_DIR):
         print(f"ERROR: Input text directory not found: {INPUT_TXT_DIR}")
         paths_ok = False

    # --- Run ---
    if paths_ok:
        try:
            process_directory(
                input_audio_dir=INPUT_AUDIO_DIR,
                input_txt_dir=INPUT_TXT_DIR,
                output_audio_dir=OUTPUT_CLIPPED_AUDIO_DIR,
                aligner_model_name=ALIGNER_MODEL,
                language=LANGUAGE,
                use_gpu=USE_GPU
            )
        except ImportError:
             pass # Exit handled earlier
        except Exception as main_e:
             print(f"\nAn unexpected error occurred during script execution: {main_e}")