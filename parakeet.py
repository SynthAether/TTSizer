import warnings
warnings.filterwarnings("ignore")

import os
import nemo.collections.asr as nemo_asr
import soundfile as sf
import torch
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import traceback
import shutil

# --- Configuration ---
SPEAKER = "Momo_Ayase"
INPUT_AUDIO_DIR = f"/home/taresh/Downloads/anime/dataset/{SPEAKER}/fil/filtered"
OUTPUT_BASE_DIR = f"/home/taresh/Downloads/anime/dataset/{SPEAKER}/fil/nemo" # Adjusted name
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
BATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TIMESTAMP_DEVIATION_THRESHOLD_SEC = 0.4
PADDING_SEC = 0.1

# Output subfolder names
FLAGGED_CROPPED_DIR_NAME = "flagged_for_review_cropped" # Contains cropped versions of flagged files
PROCESSED_MAIN_DIR_NAME = "processed_main_dataset"    # Contains all original audio + new ASR text
# --- End Configuration ---

def get_asr_boundary_times(asr_hyp_result, audio_duration_seconds):
    asr_start_sec, asr_end_sec = 0.0, audio_duration_seconds
    if hasattr(asr_hyp_result, 'timestamp') and \
       isinstance(asr_hyp_result.timestamp, dict) and \
       'word' in asr_hyp_result.timestamp and \
       asr_hyp_result.timestamp['word']:
        word_segments = asr_hyp_result.timestamp['word']
        valid_word_segments = [s for s in word_segments if isinstance(s, dict) and 'start' in s and 'end' in s]
        if valid_word_segments:
            try:
                current_start = min(s['start'] for s in valid_word_segments)
                current_end = max(s['end'] for s in valid_word_segments)
                asr_start_sec = max(0.0, current_start)
                asr_end_sec = min(audio_duration_seconds, current_end)
                if asr_end_sec < asr_start_sec:
                    asr_start_sec, asr_end_sec = 0.0, audio_duration_seconds
            except (ValueError, TypeError, KeyError): pass
    return asr_start_sec, asr_end_sec

def main():
    output_base_path = Path(OUTPUT_BASE_DIR)
    flagged_cropped_output_path = output_base_path / FLAGGED_CROPPED_DIR_NAME
    processed_main_output_path = output_base_path / PROCESSED_MAIN_DIR_NAME

    for p in [flagged_cropped_output_path, processed_main_output_path]:
        if p.exists(): shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    print(f"Loading ASR model: {ASR_MODEL_NAME} on {DEVICE}...")
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME, map_location=DEVICE)
        asr_model.eval()
    except Exception as e:
        print(f"Fatal: Error loading ASR model: {e}"); traceback.print_exc(); return
    print("ASR model loaded.")

    audio_files = sorted(list(Path(INPUT_AUDIO_DIR).glob("*.wav")))
    if not audio_files: print(f"No .wav files found in {INPUT_AUDIO_DIR}"); return
    print(f"Found {len(audio_files)} .wav files.")

    flagged_count = 0
    processed_main_count = 0 # Tracks files copied to the main processed directory

    for i in tqdm(range(0, len(audio_files), BATCH_SIZE), desc="Processing batches"):
        batch_paths_str = [str(p) for p in audio_files[i:i + BATCH_SIZE]]
        batch_paths_obj = audio_files[i:i + BATCH_SIZE]

        try:
            asr_results = asr_model.transcribe(
                batch_paths_str,
                batch_size=len(batch_paths_str),
                timestamps=True,
                verbose=False
            )
        except Exception as e:
            print(f"Error during ASR transcription: {e}"); traceback.print_exc(); continue

        if len(asr_results) != len(batch_paths_obj):
            print(f"Warning: Mismatch in ASR result count. Skipping batch."); continue

        for original_audio_path, asr_hyp in zip(batch_paths_obj, asr_results):
            try:
                full_audio_data, original_sr = sf.read(str(original_audio_path), dtype='float32')
                original_duration = len(full_audio_data) / original_sr

                asr_text_to_save = (hasattr(asr_hyp, 'text') and asr_hyp.text) or ""
                
                raw_asr_start_sec, raw_asr_end_sec = get_asr_boundary_times(asr_hyp, original_duration)

                # --- Step 1: Always copy original audio and ASR text to the main processed directory ---
                # Copy original audio
                main_output_audio_path = processed_main_output_path / original_audio_path.name
                shutil.copy2(str(original_audio_path), main_output_audio_path)
                
                # Save new ASR transcription
                main_output_txt_path = processed_main_output_path / original_audio_path.with_suffix(".txt").name
                with open(main_output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(asr_text_to_save)
                processed_main_count +=1

                # --- Step 2: Check for flagging and if flagged, also save cropped version ---
                start_deviation = abs(raw_asr_start_sec - 0.0)
                end_deviation = abs(raw_asr_end_sec - original_duration)
                
                is_flagged = start_deviation > TIMESTAMP_DEVIATION_THRESHOLD_SEC or \
                             end_deviation > TIMESTAMP_DEVIATION_THRESHOLD_SEC
                
                if is_flagged:
                    # Apply PADDING and CLAMP for flagged files before cropping
                    padded_asr_start_sec = raw_asr_start_sec - PADDING_SEC
                    padded_asr_end_sec = raw_asr_end_sec + PADDING_SEC

                    final_crop_start_sec = max(0.0, padded_asr_start_sec)
                    final_crop_end_sec = min(original_duration, padded_asr_end_sec)
                    
                    if final_crop_end_sec <= final_crop_start_sec:
                        if raw_asr_end_sec > raw_asr_start_sec:
                            final_crop_start_sec, final_crop_end_sec = raw_asr_start_sec, raw_asr_end_sec
                        else:
                            final_crop_start_sec, final_crop_end_sec = 0.0, original_duration
                            # print(f"  Note (Flagged): Both raw ASR and padded timestamps problematic for {original_audio_path.name}. Using full audio for cropped version in flagged dir.")

                    start_frame = int(final_crop_start_sec * original_sr)
                    end_frame = int(final_crop_end_sec * original_sr)

                    audio_to_save_in_flagged = None
                    if end_frame <= start_frame or start_frame < 0 or end_frame > len(full_audio_data):
                        audio_to_save_in_flagged = full_audio_data
                        # print(f"  Note (Flagged): Invalid frame range for {original_audio_path.name} in flagged dir. Using full audio.")
                    else:
                        audio_to_save_in_flagged = full_audio_data[start_frame:end_frame]
                    
                    if audio_to_save_in_flagged.size == 0 and full_audio_data.size > 0:
                        audio_to_save_in_flagged = full_audio_data
                        # print(f"  Note (Flagged): Cropped segment empty for {original_audio_path.name} in flagged dir. Using full audio.")
                    elif audio_to_save_in_flagged.size == 0 and full_audio_data.size == 0:
                        # print(f"  Warning (Flagged): Original and cropped audio empty for {original_audio_path.name}. Skipping save to flagged dir.")
                        continue # Skip saving to flagged if audio is completely empty
                    
                    # Save cropped audio to flagged directory
                    flagged_output_audio_path = flagged_cropped_output_path / original_audio_path.name
                    sf.write(str(flagged_output_audio_path), audio_to_save_in_flagged, original_sr)
                    
                    # Save ASR transcription also to flagged directory
                    flagged_output_txt_path = flagged_cropped_output_path / original_audio_path.with_suffix(".txt").name
                    with open(flagged_output_txt_path, 'w', encoding='utf-8') as f:
                        f.write(asr_text_to_save)
                    
                    flagged_count += 1
            
            except Exception as e_file:
                print(f"Error processing file {original_audio_path.name}: {e_file}")
                # traceback.print_exc() # Uncomment for detailed file error debugging

    print("\n--- Processing Complete ---")
    print(f"Total input files: {len(audio_files)}")
    print(f"  Files copied to '{PROCESSED_MAIN_DIR_NAME}' (original audio & new ASR text): {processed_main_count}")
    print(f"  Flagged files (cropped version & ASR text also saved to '{FLAGGED_CROPPED_DIR_NAME}'): {flagged_count}")
    print(f"Output base directory: {output_base_path.resolve()}")

if __name__ == "__main__":
    main()