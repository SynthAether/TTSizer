import warnings
warnings.filterwarnings("ignore")

import os
import glob
import torch
import torchaudio
import torchaudio.functional as F # For resampling
import soundfile as sf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from collections import defaultdict
import statistics
import wespeaker # Import wespeaker
import tempfile # For potential temporary resampled files
import shutil # For checking executable path

# --- Configuration ---
INPUT_DIR = 'temp/segmented_vocals_e10_final2' # Directory containing speaker subfolders with FINAL clipped WAVs

# *** WeSpeaker Configuration ***
# REQUIRED: Update this path to the directory where your downloaded WeSpeaker ResNet model files are located
# (e.g., the directory containing avg_model.pt, config.yaml)
WESPEAKER_MODEL_DIR = "/Workspace/tr/repos/wespeaker-voxceleb-resnet293-LM"

# Target sample rate expected by the WeSpeaker model (Likely 16000, CHECK MODEL DOCS)
TARGET_SR = 16000

USE_GPU = True # Set to False to force CPU

# Outlier Detection Method ('std_dev', 'percentile', 'fixed')
OUTLIER_METHOD = 'std_dev'
# Threshold:
# - For 'std_dev': Number of standard deviations from the mean distance
# - For 'percentile': The percentile (e.g., 95 means flag top 5%)
# - For 'fixed': The absolute cosine distance threshold (e.g., 0.5 - needs tuning)
OUTLIER_THRESHOLD = 2.0 # Example: Flag if distance > mean + 2.0 * std_dev
MIN_SEGMENTS_PER_SPEAKER = 5 # Min segments needed to reliably calculate centroid/stats

# --- Functions ---

def load_wespeaker_model(model_dir, use_gpu):
    """Loads the specified WeSpeaker model from a local directory."""
    print(f"Loading WeSpeaker model from: {model_dir}...")
    if not os.path.isdir(model_dir):
        print(f"Error: WeSpeaker model directory not found: {model_dir}")
        raise FileNotFoundError(f"WeSpeaker model directory not found: {model_dir}")
    try:
        model = wespeaker.load_model_local(model_dir)
        if use_gpu and torch.cuda.is_available():
            # wespeaker uses integer ID, 0 for first GPU
            gpu_id = 0
            print(f"Setting WeSpeaker model to use GPU: {gpu_id}")
            model.set_device("cuda")
        else:
            # Use -1 or other negative number for CPU
            cpu_id = -1
            print("Setting WeSpeaker model to use CPU.")
            model.set_gpu(cpu_id)
        print("WeSpeaker model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading WeSpeaker model from '{model_dir}': {e}")
        raise # Exit if model can't load

def extract_wespeaker_embedding(model, audio_path, target_sr):
    """
    Loads audio, resamples if needed, saves temporarily (if needed),
    and extracts speaker embedding using WeSpeaker.
    """
    temp_wav_path = None # Path for temporary resampled file
    try:
        signal, sr = torchaudio.load(audio_path)

        # Ensure mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Resample if necessary
        if sr != target_sr:
            # print(f"Resampling {os.path.basename(audio_path)} from {sr} Hz to {target_sr} Hz")
            signal = F.resample(signal, orig_freq=sr, new_freq=target_sr)
            sr = target_sr # Update sample rate

            # WeSpeaker's extract_embedding takes a path. If we resampled,
            # we need to save the resampled audio temporarily.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_wav_path = tmp_file.name
            # Save the resampled tensor using soundfile (requires numpy)
            sf.write(temp_wav_path, signal.squeeze().cpu().numpy(), target_sr, subtype='PCM_16') # Save as PCM_16 common format
            path_to_embed = temp_wav_path
        else:
            # No resampling needed, can use original path
            path_to_embed = audio_path

        # Extract embedding using the path
        # Assuming wespeaker model handles device internally after set_gpu
        embedding = model.extract_embedding(path_to_embed) # Returns numpy array

        if embedding is None or embedding.size == 0:
             print(f"Warning: WeSpeaker returned empty embedding for {os.path.basename(audio_path)}. Skipping.")
             return None

        if np.isnan(embedding).any():
             print(f"Warning: NaN values found in WeSpeaker embedding for {os.path.basename(audio_path)}. Skipping.")
             return None

        return embedding # Should be a NumPy array

    except Exception as e:
        print(f"Error processing/embedding file {os.path.basename(audio_path)} with WeSpeaker: {e}")
        return None
    finally:
         # Clean up temporary file if created
         if temp_wav_path and os.path.exists(temp_wav_path):
             try:
                 os.remove(temp_wav_path)
             except Exception as e_clean:
                 print(f"Warning: Could not delete temporary resampled file {temp_wav_path}: {e_clean}")


def find_speaker_outliers(input_dir, model, target_sr, outlier_method, threshold, min_segments):
    """Finds potential outlier segments for each speaker based on embedding distance."""

    all_embeddings = defaultdict(list) # Store embeddings per speaker: {'SpeakerName': [emb1, emb2...]}
    all_filepaths = defaultdict(list) # Store filepaths per speaker

    print("Scanning for speakers and WAV files...")
    # Find speaker directories directly under the input_dir
    try:
         # List items in input_dir and filter for directories
         speaker_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    except FileNotFoundError:
         print(f"Error: Input directory not found: {input_dir}")
         return {}
    except Exception as e:
         print(f"Error scanning input directory {input_dir}: {e}")
         return {}


    if not speaker_dirs:
        print(f"Error: No speaker subdirectories found in '{input_dir}'.")
        print("Expected structure: input_dir/Speaker_Name/clipped_audio.wav")
        return {}

    print(f"Found speakers: {', '.join(speaker_dirs)}")

    # --- Step 1 & 2: Extract Embeddings for all segments ---
    print("Extracting speaker embeddings (WeSpeaker) for all segments...")
    for speaker in speaker_dirs:
        speaker_path = os.path.join(input_dir, speaker)
        # Find WAV files directly in the speaker directory
        wav_files = glob.glob(os.path.join(speaker_path, '*.wav'))

        if not wav_files:
            # Check in 'wav' subdirectory as a fallback
            wav_files = glob.glob(os.path.join(speaker_path, 'wav', '*.wav'))
            if not wav_files:
                print(f"Warning: No WAV files found for speaker '{speaker}' directly in {speaker_path} or in a 'wav' subfolder.")
                continue

        print(f"  Processing {len(wav_files)} segments for speaker '{speaker}'...")
        for wav_path in tqdm(wav_files, desc=f"Embeddings for {speaker}", leave=False):
            embedding = extract_wespeaker_embedding(model, wav_path, target_sr)
            if embedding is not None:
                # Ensure embedding is 1D
                if embedding.ndim > 1:
                     embedding = embedding.flatten() # Or handle shape appropriately if needed
                all_embeddings[speaker].append(embedding)
                all_filepaths[speaker].append(wav_path)

    # --- Step 3, 4, 5: Calculate Centroids, Distances, Identify Outliers ---
    print("\nCalculating centroids and distances...")
    outliers = defaultdict(list)

    for speaker, embeddings_list in all_embeddings.items():
        num_segments = len(embeddings_list)
        print(f"  Analyzing speaker '{speaker}' ({num_segments} segments)...")

        if num_segments < min_segments:
            print(f"    Skipping outlier detection for '{speaker}': Only {num_segments} segments (minimum required: {min_segments}).")
            continue

        # Verify embedding shapes before stacking
        first_shape = embeddings_list[0].shape
        if not all(emb.shape == first_shape for emb in embeddings_list):
             print(f"    Warning: Inconsistent embedding shapes found for speaker '{speaker}'. Skipping centroid calculation.")
             # Optionally print shapes for debugging: [emb.shape for emb in embeddings_list]
             continue
        if len(first_shape) != 1:
             print(f"    Warning: Embeddings for speaker '{speaker}' are not 1D (shape: {first_shape}). Skipping centroid calculation.")
             continue

        embeddings_array = np.array(embeddings_list)
        centroid = np.mean(embeddings_array, axis=0, keepdims=True)
        similarities = cosine_similarity(centroid, embeddings_array)[0]
        distances = 1.0 - similarities
        segment_distances = list(zip(all_filepaths[speaker], distances))
        distances_only = np.array(distances)

        # Outlier Detection Logic (same as before)
        threshold_value = float('inf') # Default to no threshold
        try:
            if outlier_method == 'std_dev':
                mean_dist = np.mean(distances_only)
                std_dist = np.std(distances_only)
                if std_dist < 1e-6:
                    print(f"    Warning: Std dev near zero for '{speaker}'. Threshold effectively disabled.")
                else:
                    threshold_value = mean_dist + threshold * std_dist
                print(f"    Std Dev Method: Mean={mean_dist:.4f}, Std={std_dist:.4f}, ThresholdDist={threshold_value:.4f}")

            elif outlier_method == 'percentile':
                threshold_value = np.percentile(distances_only, threshold)
                print(f"    Percentile Method: {threshold}th Percentile Distance={threshold_value:.4f}")

            elif outlier_method == 'fixed':
                threshold_value = threshold
                print(f"    Fixed Method: Threshold Distance={threshold_value:.4f}")
            else:
                print(f"    Error: Unknown outlier method '{outlier_method}' for speaker '{speaker}'.")
                continue # Skip outlier flagging for this speaker

            # Flag outliers
            for path, dist in segment_distances:
                if dist > threshold_value:
                    outliers[speaker].append({'path': path, 'distance': float(dist)})

            outliers[speaker].sort(key=lambda x: x['distance'], reverse=True)

        except Exception as e_outlier:
             print(f"    Error during outlier calculation for '{speaker}': {e_outlier}")


    return outliers

# --- Helper Functions (_time_str_to_seconds, _sanitize_filename from previous script) ---
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

# --- Main Execution Block ---
if __name__ == "__main__":
    # Basic path check
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
    elif not os.path.isdir(WESPEAKER_MODEL_DIR):
         print(f"Error: WeSpeaker model directory not found: {WESPEAKER_MODEL_DIR}")
         print("Please download the model and update the WESPEAKER_MODEL_DIR path.")
    else:
        try:
            # Load the WeSpeaker embedding model once
            embedding_model = load_wespeaker_model(WESPEAKER_MODEL_DIR, USE_GPU)

            # Find potential outliers
            potential_outliers = find_speaker_outliers(
                INPUT_DIR,
                embedding_model,
                TARGET_SR,
                OUTLIER_METHOD,
                OUTLIER_THRESHOLD,
                MIN_SEGMENTS_PER_SPEAKER
            )

            # --- Report Results ---
            print("\n--- Potential Outlier Report (WeSpeaker) ---")
            if not potential_outliers:
                print("No potential outliers found based on the criteria.")
            else:
                total_outliers = 0
                for speaker, speaker_outliers in potential_outliers.items():
                    if speaker_outliers:
                        print(f"\nSpeaker: {speaker} ({len(speaker_outliers)} potential outliers):")
                        total_outliers += len(speaker_outliers)
                        # Print top N outliers or all if few
                        for i, outlier in enumerate(speaker_outliers):
                             if i < 15: # Show a few more potentially
                                 rel_path = os.path.relpath(outlier['path'], INPUT_DIR)
                                 print(f"  - Distance: {outlier['distance']:.4f} | File: {rel_path}")
                             elif i == 15:
                                 print(f"  ... (further outliers omitted for brevity)")
                    # Keep this part to show speakers with NO outliers found
                    # else:
                    #      print(f"\nSpeaker: {speaker}")
                    #      print(f"  No outliers found matching criteria.")


                print(f"\nTotal potential outliers flagged across all speakers: {total_outliers}")
                print("\nRecommendation: Manually review the flagged files by listening to them")
                print("to confirm if they are actually mislabeled before removing or correcting.")

        except FileNotFoundError as e:
             print(f"Error: {e}") # Handled specific path errors earlier
        except Exception as e:
            print(f"\nAn unexpected error occurred during the main process: {e}")
            import traceback
            traceback.print_exc()