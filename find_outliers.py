import warnings
warnings.filterwarnings("ignore")

import os
import glob
import torch
import soundfile as sf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import torchaudio.functional as F
from collections import defaultdict
import wespeaker
import shutil
import tempfile
import traceback

BASE_DIR = '/home/taresh/Downloads/anime/audios'

ANIME_PATHS = ['Rezero_s1/final_aligned_clips_refactored/vocals', 'Rezero_s2/final_aligned_clips_refactored/vocals', 'Rezero_s3p1/final_aligned_clips_refactored/vocals', 'Rezero_s3p2/final_aligned_clips_refactored/vocals']
OUTPUT_PATHS = ['Rezero_s1/final_aligned_clips_filtered_v2', 'Rezero_s2/final_aligned_clips_filtered_v2', 'Rezero_s3p1/final_aligned_clips_filtered_v2', 'Rezero_s3p2/final_aligned_clips_filtered_v2']

SKIP_EPISODE_PATTERNS = ['03_BD', '11_BD', '14_BD', '21_BD', 'S02E10', 'S02E01', 'S02E18', 'S03E10']
TARGET_SPEAKER_LABEL = 'Emilia'

#-----------------------------------
WESPEAKER_MODEL_DIR = "/home/taresh/Downloads/dev/TTSizer/weights/wespeaker-voxceleb-resnet293-LM"
TARGET_SR = 16000
USE_GPU = True
MIN_CLIP_DURATION_SECONDS = 0.75
CENTROID_REFINEMENT_PERCENTILE = 50
MIN_SEGMENTS_FOR_REFINEMENT = 10
OUTLIER_THRESHOLD_DEFINITE = 0.70
OUTLIER_THRESHOLD_UNCERTAIN = 0.60
MOVE_UNCERTAIN = True
UNCERTAIN_FOLDER_NAME = "uncertain"
MAX_OUTLIER_PERCENTAGE_WARN = 40.0
MIN_SEGMENTS_PER_SPEAKER_GLOBAL = 10 # For master profile from all sources
FILE_FORMAT = '*.wav'
#---------------------------------------


def should_skip_episode(episode_name, skip_patterns):
    """Checks if the episode name matches any skip patterns."""
    for pattern in skip_patterns:
        if pattern in episode_name:
            return True
    return False

def setup_output_directory_for_anime(input_path_abs, output_path_abs, target_speaker_label,
                                     skip_patterns, uncertain_folder_name, move_uncertain,
                                     file_format, base_dir_for_print):
    """
    Prepares output directory structure for an anime, copying relevant files.
    Returns a dict mapping episode names to their output speaker directory paths, or None on critical error.
    """
    input_name = os.path.basename(os.path.normpath(input_path_abs))
    output_rel_path = os.path.relpath(output_path_abs, base_dir_for_print)
    print(f"--- Setting up Output: '{input_name}' -> '{output_rel_path}' ---")

    if not os.path.isdir(input_path_abs):
         print(f"Error: Input path not found: {input_path_abs}")
         return None

    os.makedirs(output_path_abs, exist_ok=True)
    output_episode_speaker_dirs = {}

    try:
        potential_episode_dirs = [d for d in os.listdir(input_path_abs) if os.path.isdir(os.path.join(input_path_abs, d))]
        if not potential_episode_dirs:
            print(f"Info: No episode subdirectories in '{input_name}'.")
            return {}

        skipped_count = 0
        for ep_name in potential_episode_dirs:
            if should_skip_episode(ep_name, skip_patterns):
                skipped_count += 1
                continue

            input_speaker_path = os.path.join(input_path_abs, ep_name, target_speaker_label)
            if os.path.isdir(input_speaker_path):
                output_speaker_dir = os.path.join(output_path_abs, ep_name, target_speaker_label)
                filtered_path = os.path.join(output_speaker_dir, "filtered")
                outliers_path = os.path.join(output_speaker_dir, "outliers")

                os.makedirs(filtered_path, exist_ok=True)
                os.makedirs(outliers_path, exist_ok=True)
                if move_uncertain:
                    os.makedirs(os.path.join(output_speaker_dir, uncertain_folder_name), exist_ok=True)
                output_episode_speaker_dirs[ep_name] = output_speaker_dir

                files_to_copy = glob.glob(os.path.join(glob.escape(input_speaker_path), file_format))
                files_to_copy.extend(glob.glob(os.path.join(glob.escape(input_speaker_path), '*.txt')))
                if not files_to_copy: continue

                for src_file in files_to_copy:
                    dest_file = os.path.join(filtered_path, os.path.basename(src_file))
                    if not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file)
        
        if skipped_count > 0: print(f"Skipped {skipped_count} episode(s) based on patterns for '{input_name}'.")
        if not output_episode_speaker_dirs and potential_episode_dirs and skipped_count < len(potential_episode_dirs) :
            print(f"Warning: No episodes with speaker '{target_speaker_label}' processed (after skips) in '{input_name}'.")

    except Exception as e:
        print(f"Error during output setup for '{input_name}': {e}")
        traceback.print_exc()
        return None 

    print(f"--- Output Setup Complete for '{output_rel_path}' ---")
    return output_episode_speaker_dirs

def load_wespeaker_model(model_dir, use_gpu):
    """Loads the WeSpeaker model. Exits on failure."""
    print(f"Loading WeSpeaker model from: {model_dir}...")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"WeSpeaker model directory not found: {model_dir}")
    try:
        model = wespeaker.load_model_local(model_dir)
        if use_gpu and torch.cuda.is_available():
            gpu_id = 0; print(f"Setting WeSpeaker model to use GPU: {gpu_id}"); model.set_device("cuda")
        else:
            cpu_id = -1; print("Setting WeSpeaker model to use CPU."); model.set_gpu(cpu_id)
        print("WeSpeaker model loaded successfully.")
        return model
    except Exception as e:
        print(f"Fatal: Error loading WeSpeaker model: {e}")
        raise # Re-raise to be caught by main or to exit

def extract_wespeaker_embedding(model, audio_path, target_sr, min_duration):
    """Extracts WeSpeaker embedding; returns None on failure or if clip is too short."""
    try:
        info = sf.info(audio_path)
        if info.duration < min_duration:
            return None
        signal, sr = sf.read(audio_path, dtype='float32')
        signal = torch.from_numpy(np.asarray(signal))
        # if > channel is found, convert to mono
        if signal.ndim > 1 and signal.shape[-1] > 1: signal = torch.mean(signal, dim=-1)
        signal = signal.unsqueeze(0)
        # resample to targt samplerate if required
        if sr != target_sr:
            signal = F.resample(signal, orig_freq=sr, new_freq=target_sr)
            sr = target_sr
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file: temp_wav_path = tmp_file.name
            sf.write(temp_wav_path, signal.squeeze().cpu().numpy(), target_sr, subtype='PCM_16')
            path_to_embed = temp_wav_path
        else: path_to_embed = audio_path
        embedding = model.extract_embedding(path_to_embed)
        if embedding is None or embedding.size == 0 or np.isnan(embedding).any():
            return None
        return embedding.flatten()
    except Exception:
        return None

def find_target_speaker_files_in_output(output_path_abs, target_speaker_label, skip_patterns,
                                        file_format, base_dir_for_print):
    """Scans the (already setup) output directory for speaker's audio files."""
    episode_data = {}
    all_speaker_files = []
    output_rel_path = os.path.relpath(output_path_abs, base_dir_for_print)

    try:
        potential_episode_dirs = [d for d in os.listdir(output_path_abs) if os.path.isdir(os.path.join(output_path_abs, d))]
        if not potential_episode_dirs: return {}, []

        for ep_name in potential_episode_dirs:
            if should_skip_episode(ep_name, skip_patterns): continue
            filtered_dir = os.path.join(output_path_abs, ep_name, target_speaker_label, "filtered")
            if os.path.isdir(filtered_dir):
                audio_files = sorted(glob.glob(os.path.join(glob.escape(filtered_dir), file_format)))
                if audio_files:
                    episode_data[ep_name] = {
                        'speaker_dir': os.path.join(output_path_abs, ep_name, target_speaker_label),
                        'filtered_dir': filtered_dir,
                        'initial_audio_files': audio_files
                    }
                    all_speaker_files.extend(audio_files)
    except Exception as e:
         print(f"Error scanning output directory '{output_rel_path}': {e}")
         return {}, []

    return episode_data, all_speaker_files

def calculate_global_speaker_profile(model, audio_paths, target_sr, min_duration,
                                     min_segments, refine_percentile, min_refine_segments, profile_name):
    """Calculates a speaker profile (centroid) from a list of audio paths."""
    print(f"\nCalculating profile for '{profile_name}' using {len(audio_paths)} files...")
    valid_embeddings_data = [] # List of {'path': path, 'embedding': embedding}
    for f_path in tqdm(audio_paths, desc=f"Embeddings for {profile_name}"):
        embedding = extract_wespeaker_embedding(model, f_path, target_sr, min_duration)
        if embedding is not None:
            valid_embeddings_data.append({'path': f_path, 'embedding': embedding})

    num_valid = len(valid_embeddings_data)
    print(f"Extracted {num_valid} valid embeddings (after duration filter).")
    if num_valid < min_segments:
        print(f"Error: Only {num_valid} valid segments for '{profile_name}' (min required: {min_segments}). Cannot calculate profile.")
        return None, None, None

    embeddings_list = [item['embedding'] for item in valid_embeddings_data]
    if not all(emb.shape == embeddings_list[0].shape and len(emb.shape) == 1 for emb in embeddings_list):
        print(f"Error: Inconsistent/non-1D embeddings for '{profile_name}'. Cannot calculate profile.")
        return None, None, None
    
    embeddings_array = np.array(embeddings_list)
    initial_centroid = np.mean(embeddings_array, axis=0, keepdims=True)
    final_centroid = initial_centroid

    # Centroid Refinement
    if embeddings_array.shape[0] > 1: # Refinement needs more than one embedding
        distances = 1.0 - cosine_similarity(initial_centroid, embeddings_array)[0]
        if len(distances) > 0:
            dist_thresh = np.percentile(distances, refine_percentile)
            confident_indices = np.where(distances <= dist_thresh)[0]
            if len(confident_indices) >= min_refine_segments:
                final_centroid = np.mean(embeddings_array[confident_indices], axis=0, keepdims=True)
                print(f"  Refined centroid for '{profile_name}' using {len(confident_indices)} embeddings.")

    return final_centroid.flatten(), None, valid_embeddings_data # Returning None for stats for now

def identify_and_move_outliers_in_episode(
    _ep_name_for_log, # Episode name, mainly for context if logging errors here
    ep_speaker_dir,   # Abs path to .../EpisodeName/SpeakerName/
    ep_files_embeddings, # List of {'path': abs_audio_path, 'embedding': ...}
    master_centroid,
    outlier_cfg # Dict with 'definite_threshold', 'uncertain_threshold', etc.
    ):
    """Identifies and moves outliers for a single episode using the master centroid."""
    filtered_dir = os.path.join(ep_speaker_dir, "filtered")
    outliers_dir = os.path.join(ep_speaker_dir, "outliers")
    uncertain_dir = os.path.join(ep_speaker_dir, outlier_cfg['uncertain_folder_name'])

    if outlier_cfg['move_uncertain']: os.makedirs(uncertain_dir, exist_ok=True)

    moved_def, moved_unc = 0, 0
    def_report, unc_report = [], []
    
    # Ensure uncertain threshold is stricter or equal to definite
    current_uncertain_thresh = min(outlier_cfg['uncertain_threshold'], outlier_cfg['definite_threshold'] - 0.001)

    master_centroid_2d = master_centroid.reshape(1, -1)
    for item in ep_files_embeddings:
        audio_path, embedding = item['path'], item['embedding'].reshape(1, -1)
        distance = 1.0 - cosine_similarity(master_centroid_2d, embedding)[0][0]

        target_move_dir, report_list_ref, is_def, is_unc = None, None, False, False

        if distance > outlier_cfg['definite_threshold']:
            target_move_dir, report_list_ref, is_def = outliers_dir, def_report, True
        elif distance > current_uncertain_thresh:
            report_list_ref = unc_report
            if outlier_cfg['move_uncertain']:
                target_move_dir, is_unc = uncertain_dir, True
        
        if target_move_dir: # File to be moved
            base_name = os.path.basename(audio_path)
            txt_path = os.path.join(filtered_dir, os.path.splitext(base_name)[0] + ".txt")
            try:
                shutil.move(audio_path, os.path.join(target_move_dir, base_name))
                if os.path.exists(txt_path): shutil.move(txt_path, os.path.join(target_move_dir, os.path.basename(txt_path)))
                report_list_ref.append({'path': audio_path, 'distance': float(distance)})
                if is_def: moved_def += 1
                elif is_unc: moved_unc += 1
            except Exception as e:
                print(f"    Error moving {base_name} to {os.path.relpath(target_move_dir, ep_speaker_dir)}: {e}")
        elif report_list_ref is not None: # Uncertain, but not moving
            report_list_ref.append({'path': audio_path, 'distance': float(distance)})

    def_report.sort(key=lambda x: x['distance'], reverse=True)
    unc_report.sort(key=lambda x: x['distance'], reverse=True)
    return {"definite": def_report, "uncertain": unc_report}, moved_def, moved_unc

def print_anime_outlier_summary(input_frag, output_frag, output_abs,
                                total_def_moved, total_unc_moved, all_ep_reports,
                                ep_file_data_for_anime, move_uncertain_flag, uncertain_f_name):
    """Prints a summary of outlier processing for a single anime source."""
    print(f"\n--- Outlier Summary for Anime Source: {input_frag} ---")
    print(f"Output Path: {output_frag} (at {output_abs})")
    
    total_unc_reported = sum(len(d['uncertain']) for d in all_ep_reports.values() if not move_uncertain_flag)

    if total_def_moved == 0 and total_unc_moved == 0 and total_unc_reported == 0:
        print("No definite or uncertain outliers were moved or reported.")
        return

    print(f"Total DEFINITE files moved: {total_def_moved}")
    if move_uncertain_flag: print(f"Total UNCERTAIN files moved to '{uncertain_f_name}': {total_unc_moved}")
    else: print(f"Total UNCERTAIN files reported (not moved): {total_unc_reported}")

    print("\nMoved/Reported File Details (Top 5 per category per episode):")
    for ep_name, reports in sorted(all_ep_reports.items()):
        if not reports.get('definite') and not reports.get('uncertain'): continue
        print(f"  Episode: {ep_name}")
        ep_meta = ep_file_data_for_anime.get(ep_name)
        if not ep_meta: continue # Should have metadata if reports exist

        original_filt_dir = ep_meta['filtered_dir']
        for category, cat_reports in reports.items():
            if not cat_reports: continue
            action = "moved" if (category == 'definite' or move_uncertain_flag) else "reported"
            print(f"    {category.capitalize()} Outliers ({len(cat_reports)} {action}):")
            for item in cat_reports[:5]:
                rel_path = os.path.relpath(item['path'], original_filt_dir)
                print(f"      - Dist: {item['distance']:.4f} | File: {rel_path}")
            if len(cat_reports) > 5: print("      ...")

def print_episode_outlier_rate_check(input_frag, ep_file_data_for_anime,
                                     all_ep_reports, max_warn_percent):
    """Checks and prints episodes with high outlier rates for an anime source."""
    print(f"\n--- Episode Outlier Rate Check for: {input_frag} (Threshold: >{max_warn_percent:.1f}%) ---")
    high_rate_eps = []
    for ep_name, ep_data in ep_file_data_for_anime.items():
        initial_files = len(ep_data.get('initial_audio_files', []))
        if initial_files == 0: continue

        num_def = len(all_ep_reports[ep_name].get('definite', []))
        num_unc = len(all_ep_reports[ep_name].get('uncertain', []))
        total_flagged = num_def + num_unc
        
        percent_flagged = (total_flagged / initial_files) * 100
        if percent_flagged > max_warn_percent:
            print(f"  - WARNING: Ep '{ep_name}' high rate: {percent_flagged:.1f}% ({total_flagged}/{initial_files})")
            high_rate_eps.append(ep_name)

    if not high_rate_eps: print("No episodes exceeded the high outlier rate warning threshold.")
    else: print("\nRecommendation: Manually review high-rate episodes listed above.")

def initialize_program_and_load_model(base_dir_val, model_dir_val, speaker_label_val,
                                      anime_paths_list, output_paths_list, use_gpu_flag):
    """Validates essential configurations and loads the embedding model."""
    if not os.path.isdir(base_dir_val): exit(f"Fatal: Base directory not found: {base_dir_val}")
    if not speaker_label_val: exit("Fatal: TARGET_SPEAKER_LABEL cannot be empty.")
    if not anime_paths_list or not output_paths_list: exit("Fatal: ANIME_PATHS or OUTPUT_PATHS list cannot be empty.")
    if len(anime_paths_list) != len(output_paths_list):
        exit(f"Fatal: ANIME_PATHS ({len(anime_paths_list)}) and OUTPUT_PATHS ({len(output_paths_list)}) must have the same number of entries.")
    
    print("Program Initializing...")
    return load_wespeaker_model(model_dir_val, use_gpu_flag)

def create_master_profile(embedding_model, anime_paths_list, output_paths_list, main_cfg):
    """
    Phase 1: Sets up directories, aggregates speaker files from all anime sources,
    and calculates a single master speaker profile.
    Returns: master_centroid, master_embedding_map, per_anime_phase2_data (dict)
    """
    print("\n=================================================")
    print("Phase 1: Master Profile Creation")
    print("=================================================")
    all_files_for_master = []
    per_anime_data_for_phase2 = {} # Stores data needed for Phase 2, keyed by input_path_fragment

    for i, input_f in enumerate(anime_paths_list):
        output_f = output_paths_list[i]
        anime_name_short = os.path.basename(os.path.normpath(input_f))
        print(f"\n--- Preparing '{anime_name_short}' for master profile ---")

        input_abs_path = os.path.join(main_cfg['base_dir'], input_f)
        output_abs_path = os.path.join(main_cfg['base_dir'], output_f)
        
        current_anime_info = {'name': anime_name_short, 'output_path_abs': output_abs_path,
                              'output_fragment': output_f, 'episode_file_data': None, 'status': 'pending'}
        per_anime_data_for_phase2[input_f] = current_anime_info

        if not os.path.isdir(input_abs_path):
            print(f"Warning: Input path '{input_f}' not found. Skipping its contribution.")
            current_anime_info['status'] = 'input_not_found'
            continue
        if os.path.abspath(input_abs_path) == os.path.abspath(output_abs_path):
            print(f"Error: Input and Output paths are same for '{input_f}'. Skipping.")
            current_anime_info['status'] = 'paths_same'
            continue

        # Setup output dirs & copy files. setup_output_directory_for_anime returns dict or None on error.
        _ = setup_output_directory_for_anime(
            input_abs_path, output_abs_path, main_cfg['speaker_label'], main_cfg['skip_patterns'],
            main_cfg['uncertain_folder_name'], main_cfg['outlier_config']['move_uncertain'], # Pass move_uncertain correctly
            main_cfg['file_format'], main_cfg['base_dir']
        )
        # We rely on find_target_speaker_files_in_output to confirm if files are usable.

        ep_data, anime_target_files_list = find_target_speaker_files_in_output(
            output_abs_path, main_cfg['speaker_label'], main_cfg['skip_patterns'],
            main_cfg['file_format'], main_cfg['base_dir']
        )
        current_anime_info['episode_file_data'] = ep_data # Store for Phase 2

        if not anime_target_files_list:
            print(f"No relevant files from '{anime_name_short}' for master profile after setup/scan.")
            current_anime_info['status'] = 'no_files_in_output' # Still might have empty dirs from setup
        else:
            all_files_for_master.extend(anime_target_files_list)
            current_anime_info['status'] = 'success_files_found'
            print(f"Added {len(anime_target_files_list)} files from '{anime_name_short}' to master profile list.")

    if not all_files_for_master:
        exit("Fatal: No speaker audio files found across all sources. Cannot create master profile.")

    master_c, _, master_embed_data = calculate_global_speaker_profile(
        embedding_model, all_files_for_master, main_cfg['target_sr'],
        main_cfg['min_clip_duration'], main_cfg['min_segments_global'],
        main_cfg['centroid_refinement_percentile'], main_cfg['min_refinement_segments'],
        "MASTER PROFILE (ALL SOURCES)"
    )
    if master_c is None:
        exit("Fatal: Could not calculate master speaker profile. Aborting.")
    
    master_embed_map = {item['path']: item['embedding'] for item in master_embed_data}
    print("Master speaker profile and embedding map created.")
    return master_c, master_embed_map, per_anime_data_for_phase2

def process_outliers_for_all_animes(master_c, master_embed_map, per_anime_data_map, main_cfg):
    """
    Phase 2: Iterates through each anime source, identifies outliers
    using the master profile, and generates reports.
    """
    print("\n=================================================")
    print("Phase 2: Per-Anime Outlier Detection (using Master Profile)")
    print("=================================================")

    for input_f, anime_info in per_anime_data_map.items():
        anime_name_short = anime_info['name']
        output_f = anime_info['output_fragment']
        output_abs_path = anime_info['output_path_abs']
        
        print(f"\n-------------------------------------------------")
        print(f"Processing Outliers for Anime: {anime_name_short} (Input: {input_f})")
        print(f"Corresponding Output Path    : {output_f}")
        print(f"-------------------------------------------------")

        if anime_info['status'] in ['input_not_found', 'paths_same', 'pending']:
            print(f"Skipping outlier detection for '{anime_name_short}' due to earlier issue: {anime_info['status']}.")
            continue
        
        ep_file_data = anime_info['episode_file_data']
        if not ep_file_data: # No episodes found for this speaker, or all skipped, or setup failed to return data
            print(f"No processable episode data for '{anime_name_short}'. Skipping outlier detection.")
            continue

        # Check if any files from THIS anime have embeddings in the master map
        files_from_this_anime_with_embeds = [
            f for ep_d_list in ep_file_data.values() for f in ep_d_list.get('initial_audio_files', []) if f in master_embed_map
        ]
        if not files_from_this_anime_with_embeds:
            print(f"No audio files from '{anime_name_short}' have valid embeddings in master map. Skipping its outlier detection.")
            continue

        print(f"\nIdentifying outliers in '{output_f}' using MASTER profile...")
        anime_total_def_moved, anime_total_unc_moved = 0, 0
        anime_all_ep_reports = defaultdict(lambda: {"definite": [], "uncertain": []})

        for ep_name, ep_d in tqdm(ep_file_data.items(), desc=f"Episodes in {output_f}"):
            ep_files_to_check = [{'path': f_p, 'embedding': master_embed_map[f_p]}
                                 for f_p in ep_d.get('initial_audio_files', []) if f_p in master_embed_map]
            if not ep_files_to_check: continue

            reports, moved_d, moved_u = identify_and_move_outliers_in_episode(
                ep_name, ep_d['speaker_dir'], ep_files_to_check,
                master_c, main_cfg['outlier_config'] # Pass the outlier_config sub-dictionary
            )
            anime_total_def_moved += moved_d
            anime_total_unc_moved += moved_u
            anime_all_ep_reports[ep_name] = reports
        
        print_anime_outlier_summary(
            input_f, output_f, output_abs_path,
            anime_total_def_moved, anime_total_unc_moved, anime_all_ep_reports,
            ep_file_data, main_cfg['outlier_config']['move_uncertain'], main_cfg['outlier_config']['uncertain_folder_name']
        )
        print_episode_outlier_rate_check(
            input_f, ep_file_data, anime_all_ep_reports, main_cfg['max_outlier_percentage_warn']
        )
        print(f"\n--- Finished outlier processing for Anime: {anime_name_short} ---")

def main():
    """Main function to orchestrate the audio outlier detection process."""
    # Central configuration dictionary
    config = {
        'base_dir': BASE_DIR,
        'wespeaker_model_dir': WESPEAKER_MODEL_DIR,
        'target_sr': TARGET_SR,
        'use_gpu': USE_GPU,
        'min_clip_duration': MIN_CLIP_DURATION_SECONDS,
        'centroid_refinement_percentile': CENTROID_REFINEMENT_PERCENTILE,
        'min_refinement_segments': MIN_SEGMENTS_FOR_REFINEMENT,
        'min_segments_global': MIN_SEGMENTS_PER_SPEAKER_GLOBAL,
        'file_format': FILE_FORMAT,
        'speaker_label': TARGET_SPEAKER_LABEL,
        'skip_patterns': SKIP_EPISODE_PATTERNS,
        'uncertain_folder_name': UNCERTAIN_FOLDER_NAME,
        'max_outlier_percentage_warn': MAX_OUTLIER_PERCENTAGE_WARN,
        'outlier_config': { # Sub-dictionary for outlier-specific settings
            'definite_threshold': OUTLIER_THRESHOLD_DEFINITE,
            'uncertain_threshold': OUTLIER_THRESHOLD_UNCERTAIN,
            'move_uncertain': MOVE_UNCERTAIN,
            'uncertain_folder_name': UNCERTAIN_FOLDER_NAME # Access within identify_and_move
        }
    }

    try:
        embedding_model = initialize_program_and_load_model(
            config['base_dir'], config['wespeaker_model_dir'], config['speaker_label'],
            ANIME_PATHS, OUTPUT_PATHS, config['use_gpu']
        )

        master_centroid, master_embedding_map, per_anime_data = create_master_profile(
            embedding_model, ANIME_PATHS, OUTPUT_PATHS, config
        )

        process_outliers_for_all_animes(
            master_centroid, master_embedding_map, per_anime_data, config
        )

    except FileNotFoundError as e: # Specific common error
        print(f"Fatal File Not Found Error: {e}")
        exit(1)
    except Exception as e: # Catch-all for other critical issues
        print(f"\nAn unexpected critical error occurred: {e}")
        traceback.print_exc()
        exit(1)

    print("\n=================================================")
    print("All specified anime sources processed.")
    print("=================================================")

if __name__ == "__main__":
    main()