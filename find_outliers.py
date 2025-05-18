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
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv

class OutlierDetector:
    """Identifies and moves audio clips that are outliers to a target speaker's voice profile."""

    def __init__(self, global_config: Dict[str, Any], outlier_config: Dict[str, Any]):
        # self.project_root_dir = Path(global_config["project_settings"]["project_root_dir"]) # REMOVED
        # self.processing_output_parent_dir = Path(global_config["project_settings"]["processing_output_parent_dir"]) # REMOVED
        
        # Resolve speaker_embedding_model_path relative to CWD (project root)
        # Assumes CWD is set to project root by orchestrator or standalone script
        speaker_model_rel_path = global_config['global_model_paths']['speaker_embedding_model']
        self.speaker_embedding_model_path = Path(speaker_model_rel_path)
        if not self.speaker_embedding_model_path.is_absolute():
            self.speaker_embedding_model_path = Path.cwd() / self.speaker_embedding_model_path
        self.speaker_embedding_model_path = self.speaker_embedding_model_path.resolve()
        
        # Outlier detection specific parameters
        self.target_sr: int = outlier_config["target_sample_rate"]
        self.use_gpu: bool = outlier_config["use_gpu"]
        self.min_clip_duration_seconds: float = outlier_config["min_clip_duration_seconds"]
        self.centroid_refinement_percentile: int = outlier_config["centroid_refinement_percentile"]
        self.min_segments_for_refinement: int = outlier_config["min_segments_for_refinement"]
        self.min_segments_for_master_profile: int = outlier_config["min_segments_for_master_profile"]
        
        self.definite_threshold: float = outlier_config["outlier_threshold_definite"]
        self.uncertain_threshold: float = outlier_config["outlier_threshold_uncertain"]
        self.move_uncertain: bool = outlier_config["move_uncertain_files"]
        self.uncertain_folder_name: str = outlier_config["uncertain_folder_name"]
        self.max_outlier_percentage_warn: float = outlier_config["max_outlier_percentage_warn"]
        
        self.file_format_glob: str = outlier_config.get("audio_file_format_glob", "*.wav") 
        self.skip_episode_patterns: List[str] = outlier_config.get("skip_episode_patterns", [])
        
        self.embedding_model: Optional[wespeaker.Speaker] = self._load_embedding_model()
        tqdm.write("OutlierDetector initialized.")

    def _load_embedding_model(self) -> Optional[wespeaker.Speaker]:
        tqdm.write(f"OutlierDetector: Loading speaker embedding model from: {self.speaker_embedding_model_path}")
        if not self.speaker_embedding_model_path.is_dir():
            raise FileNotFoundError(f"Speaker embedding model directory not found: {self.speaker_embedding_model_path}")
        try:
            model = wespeaker.load_model_local(str(self.speaker_embedding_model_path))
            if self.use_gpu and torch.cuda.is_available():
                model.set_device("cuda") 
                tqdm.write("OutlierDetector: Speaker embedding model set to use GPU.")
            else:
                model.set_device("cpu") 
                tqdm.write("OutlierDetector: Speaker embedding model set to use CPU.")
            return model
        except Exception as e:
            raise RuntimeError(f"Fatal: Error loading WeSpeaker model: {e}")

    def _should_skip_episode(self, episode_name: str) -> bool:
        return any(pattern in episode_name for pattern in self.skip_episode_patterns)

    def _extract_embedding(self, audio_path: Path) -> Optional[np.ndarray]:
        try:
            info = sf.info(str(audio_path))
            if info.duration < self.min_clip_duration_seconds:
                return None
            
            signal, sr = sf.read(str(audio_path), dtype='float32')
            signal_tensor = torch.from_numpy(np.asarray(signal))

            if signal_tensor.ndim > 1 and signal_tensor.shape[-1] > 1: 
                signal_tensor = torch.mean(signal_tensor, dim=-1)
            if signal_tensor.ndim == 1:
                signal_tensor = signal_tensor.unsqueeze(0) 

            path_for_embedding = str(audio_path)
            temp_file_to_delete = None

            if sr != self.target_sr:
                signal_tensor = F.resample(signal_tensor, orig_freq=sr, new_freq=self.target_sr)
                sr = self.target_sr
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                    temp_file_to_delete = Path(tmp_f.name)
                sf.write(str(temp_file_to_delete), signal_tensor.squeeze().cpu().numpy(), self.target_sr, subtype='PCM_16')
                path_for_embedding = str(temp_file_to_delete)
            
            embedding = self.embedding_model.extract_embedding(path_for_embedding)
            
            if temp_file_to_delete and temp_file_to_delete.exists():
                temp_file_to_delete.unlink()

            if embedding is None or embedding.size == 0 or np.isnan(embedding).any():
                return None
            return embedding.flatten()
        except Exception as e:
            # tqdm.write(f"    Warning: Could not extract embedding for {audio_path.name}: {e}")
            return None

    def _setup_initial_filtered_dirs_and_copy_files(
        self, project_name: str, source_config: Dict[str, Any], 
        input_base_dir: Path, output_base_dir: Path
    ) -> Dict[str, Dict[str, Any]]:
        source_input_subpath = Path(source_config["input_subpath_relative_to_input_stage"])
        target_speaker_label = source_config["target_speaker_label"]
        source_actual_input_dir = input_base_dir / source_input_subpath
        source_current_stage_output_base = output_base_dir / source_input_subpath
        
        tqdm.write(f"  Setting up output for source: {source_input_subpath} (Speaker: {target_speaker_label})")
        tqdm.write(f"    Input from: {source_actual_input_dir}")
        tqdm.write(f"    Base output for this source in current stage: {source_current_stage_output_base}")

        if not source_actual_input_dir.is_dir():
            tqdm.write(f"    Warning: Source input path not found: {source_actual_input_dir}. Skipping this source.")
            return {}

        output_episode_data_map = {}
        for ep_dir in source_actual_input_dir.iterdir():
            if not ep_dir.is_dir() or self._should_skip_episode(ep_dir.name):
                if ep_dir.is_dir(): tqdm.write(f"    Skipping episode by pattern: {ep_dir.name}")
                continue

            output_episode_speaker_dir = source_current_stage_output_base / ep_dir.name
            filtered_path = output_episode_speaker_dir / "filtered"
            outliers_path = output_episode_speaker_dir / "outliers"
            
            filtered_path.mkdir(parents=True, exist_ok=True)
            outliers_path.mkdir(parents=True, exist_ok=True)
            if self.move_uncertain:
                (output_episode_speaker_dir / self.uncertain_folder_name).mkdir(parents=True, exist_ok=True)

            copied_audio_files = []
            files_in_input_ep_dir = list(ep_dir.glob(f"*{self.file_format_glob[-4:]}")) 
            files_in_input_ep_dir.extend(list(ep_dir.glob("*.txt"))) 

            if not files_in_input_ep_dir:
                continue
            
            for src_file in files_in_input_ep_dir:
                dest_file = filtered_path / src_file.name
                if not dest_file.exists(): 
                    shutil.copy2(src_file, dest_file)
                if src_file.suffix.lower() == self.file_format_glob[-4:].lower():
                    copied_audio_files.append(dest_file) 

            if copied_audio_files:
                 output_episode_data_map[ep_dir.name] = {
                    'speaker_dir_current_stage': output_episode_speaker_dir,
                    'filtered_dir_current_stage': filtered_path, 
                    'initial_audio_files': sorted(copied_audio_files) 
                }
        return output_episode_data_map

    def _calculate_speaker_profile(self, all_audio_paths: List[Path], profile_name_for_log: str) -> Tuple[Optional[np.ndarray], Dict[Path, np.ndarray]]:
        tqdm.write(f"\nCalculating profile for '{profile_name_for_log}' using {len(all_audio_paths)} candidate files...")
        embeddings_map: Dict[Path, np.ndarray] = {}
        temp_embeddings_list_for_np = []

        for f_path in tqdm(all_audio_paths, desc=f"Embedding for {profile_name_for_log}", leave=False, unit="file"):
            embedding = self._extract_embedding(f_path)
            if embedding is not None:
                embeddings_map[f_path] = embedding
                temp_embeddings_list_for_np.append(embedding)

        num_valid_embeddings = len(temp_embeddings_list_for_np)
        tqdm.write(f"  Extracted {num_valid_embeddings} valid embeddings for '{profile_name_for_log}' (after duration filter)." )
        
        if num_valid_embeddings < self.min_segments_for_master_profile:
            tqdm.write(f"  ERROR: Only {num_valid_embeddings} valid segments for '{profile_name_for_log}' (min required: {self.min_segments_for_master_profile}). Cannot calculate profile.")
            return None, embeddings_map

        first_emb_shape = temp_embeddings_list_for_np[0].shape
        if not all(emb.shape == first_emb_shape and len(emb.shape) == 1 for emb in temp_embeddings_list_for_np):
            tqdm.write(f"  ERROR: Inconsistent or non-1D embeddings for '{profile_name_for_log}'. Cannot calculate profile.")
            return None, embeddings_map

        embeddings_array = np.array(temp_embeddings_list_for_np)
        initial_centroid = np.mean(embeddings_array, axis=0, keepdims=True)
        final_centroid = initial_centroid.flatten()

        if embeddings_array.shape[0] >= self.min_segments_for_refinement:
            distances = 1.0 - cosine_similarity(initial_centroid, embeddings_array)[0]
            if len(distances) > 0 :
                dist_thresh = np.percentile(distances, self.centroid_refinement_percentile)
                confident_indices = np.where(distances <= dist_thresh)[0]
                if len(confident_indices) >= self.min_segments_for_refinement:
                    final_centroid = np.mean(embeddings_array[confident_indices], axis=0)
                    tqdm.write(f"  Refined centroid for '{profile_name_for_log}' using {len(confident_indices)} embeddings (top {100-self.centroid_refinement_percentile}th percentile by similarity).")
        else:
            tqdm.write(f"  Skipping centroid refinement for '{profile_name_for_log}', not enough embeddings ({embeddings_array.shape[0]} < {self.min_segments_for_refinement}).")
            
        return final_centroid, embeddings_map

    def _identify_and_move_outliers_for_episode(
        self, episode_name: str, episode_speaker_dir_current_stage: Path, 
        episode_audio_files_with_embeddings: List[Dict[str, Any]],
        master_centroid: np.ndarray
    ) -> Tuple[Dict[str, List[Dict]], int, int]:
        
        filtered_dir = episode_speaker_dir_current_stage / "filtered"
        outliers_dir = episode_speaker_dir_current_stage / "outliers"
        uncertain_dir = episode_speaker_dir_current_stage / self.uncertain_folder_name

        moved_definite_count, moved_uncertain_count = 0, 0
        report_definite: List[Dict] = []
        report_uncertain: List[Dict] = []
        
        current_uncertain_thresh = min(self.uncertain_threshold, self.definite_threshold - 0.001)
        master_centroid_2d = master_centroid.reshape(1, -1)

        for item in episode_audio_files_with_embeddings:
            audio_path: Path = item['path']
            embedding: np.ndarray = item['embedding'].reshape(1, -1)
            
            if not audio_path.exists():
                continue

            distance = 1.0 - cosine_similarity(master_centroid_2d, embedding)[0][0]
            target_move_dir: Optional[Path] = None
            report_list_ref: Optional[List[Dict]] = None
            is_definite_outlier, is_uncertain_outlier = False, False

            if distance > self.definite_threshold:
                target_move_dir, report_list_ref, is_definite_outlier = outliers_dir, report_definite, True
            elif distance > current_uncertain_thresh:
                report_list_ref = report_uncertain
                if self.move_uncertain:
                    target_move_dir, is_uncertain_outlier = uncertain_dir, True
            
            if target_move_dir:
                base_name = audio_path.name
                txt_name = audio_path.with_suffix(".txt").name
                original_txt_path = filtered_dir / txt_name
                
                try:
                    shutil.move(str(audio_path), str(target_move_dir / base_name))
                    if original_txt_path.exists():
                        shutil.move(str(original_txt_path), str(target_move_dir / txt_name))
                    
                    if report_list_ref is not None:
                        report_list_ref.append({'path': audio_path, 'distance': float(distance)})
                    
                    if is_definite_outlier: moved_definite_count += 1
                    elif is_uncertain_outlier: moved_uncertain_count += 1
                except Exception as e:
                    tqdm.write(f"    Error moving {base_name} for ep {episode_name} to {target_move_dir.name}: {e}")
            elif report_list_ref is not None: 
                report_list_ref.append({'path': audio_path, 'distance': float(distance)})

        report_definite.sort(key=lambda x: x['distance'], reverse=True)
        report_uncertain.sort(key=lambda x: x['distance'], reverse=True)
        return {"definite": report_definite, "uncertain": report_uncertain}, moved_definite_count, moved_uncertain_count

    def _print_source_summary(self, source_input_display_name: str, 
                             source_output_base_current_stage: Path,
                             total_def_moved: int, total_unc_moved: int, 
                             all_ep_reports: Dict[str, Dict[str, List[Dict]]],
                             episode_processing_details_map: Dict[str, Dict[str, Any]]):
        tqdm.write(f"\n--- Outlier Summary for Source: {source_input_display_name} ---")
        tqdm.write(f"  Output Path (this stage): {source_output_base_current_stage.resolve()}")
        
        total_unc_reported_not_moved = 0
        if not self.move_uncertain:
            total_unc_reported_not_moved = sum(len(r.get('uncertain',[])) for r in all_ep_reports.values())

        if total_def_moved == 0 and total_unc_moved == 0 and total_unc_reported_not_moved == 0:
            tqdm.write("  No definite or uncertain outliers were moved or reported for this source.")
            return

        tqdm.write(f"  Total DEFINITE files moved to 'outliers': {total_def_moved}")
        if self.move_uncertain:
            tqdm.write(f"  Total UNCERTAIN files moved to '{self.uncertain_folder_name}': {total_unc_moved}")
        else:
            tqdm.write(f"  Total UNCERTAIN files reported (not moved): {total_unc_reported_not_moved}")

        if total_def_moved > 0 or total_unc_moved > 0 or total_unc_reported_not_moved > 0:
            tqdm.write("\n  Moved/Reported File Details (Top 5 per category per episode):")
            for ep_name, reports in sorted(all_ep_reports.items()):
                if not reports.get('definite') and not reports.get('uncertain'):
                    continue
                tqdm.write(f"    Episode: {ep_name}")
                ep_details = episode_processing_details_map.get(ep_name)
                if not ep_details: continue

                for category, cat_reports in reports.items():
                    if not cat_reports: continue
                    action = "moved" if (category == 'definite' or self.move_uncertain) else "reported"
                    tqdm.write(f"      {category.capitalize()} Outliers ({len(cat_reports)} {action}):")
                    for item in cat_reports[:5]:
                        display_path = item['path'].name 
                        tqdm.write(f"        - Dist: {item['distance']:.4f} | File: {display_path}")
                    if len(cat_reports) > 5: tqdm.write("        ...")
        tqdm.write(f"--- End Summary for Source: {source_input_display_name} ---")

    def _check_episode_outlier_rates(self, source_input_display_name: str, 
                                    episode_processing_details_map: Dict[str, Dict[str, Any]],
                                    all_ep_reports: Dict[str, Dict[str, List[Dict]]]):
        tqdm.write(f"\n--- Episode Outlier Rate Check for: {source_input_display_name} (Threshold: >{self.max_outlier_percentage_warn:.1f}%) ---")
        high_rate_eps_found = False
        for ep_name, ep_details in episode_processing_details_map.items():
            initial_files_count = len(ep_details.get('initial_audio_files', []))
            if initial_files_count == 0:
                continue

            ep_report_data = all_ep_reports.get(ep_name, {})
            num_def = len(ep_report_data.get('definite', []))
            num_unc = len(ep_report_data.get('uncertain', [])) 
            total_flagged = num_def + num_unc
            
            percent_flagged = (total_flagged / initial_files_count) * 100 if initial_files_count > 0 else 0
            if percent_flagged > self.max_outlier_percentage_warn:
                tqdm.write(f"  - WARNING: Ep '{ep_name}' high outlier rate: {percent_flagged:.1f}% ({total_flagged}/{initial_files_count} flagged)")
                high_rate_eps_found = True

        if not high_rate_eps_found:
            tqdm.write("  No episodes exceeded the high outlier rate warning threshold for this source.")
        else:
            tqdm.write("\n  Recommendation: Manually review high-rate episodes listed above.")
        tqdm.write(f"--- End Rate Check for Source: {source_input_display_name} ---")

    def run_outlier_identification_for_project(
        self, project_name: str, 
        project_audio_sources_config: List[Dict[str, Any]],
        project_input_base_dir: Path,
        project_output_base_dir: Path
    ):
        if not self.embedding_model:
            tqdm.write("OutlierDetector ERROR: Speaker embedding model not loaded. Aborting.")
            return

        tqdm.write("\n" + "="*60)
        tqdm.write(f"OutlierDetector: Starting Outlier Identification for Project: '{project_name}'")
        tqdm.write(f"  Input base directory: '{project_input_base_dir.resolve()}'")
        tqdm.write(f"  Output base directory: '{project_output_base_dir.resolve()}'")
        tqdm.write("="*60 + "\n")

        project_output_base_dir.mkdir(parents=True, exist_ok=True)

        tqdm.write("--- Phase 1: Master Profile Creation ---")
        all_files_for_master_profile: List[Path] = []
        per_source_processing_data: Dict[str, Dict[str, Any]] = {} 

        for source_conf in project_audio_sources_config:
            input_subpath_str = source_conf["input_subpath_relative_to_input_stage"]
            speaker_label = source_conf["target_speaker_label"]
            source_display_name = f"{input_subpath_str} (Speaker: {speaker_label})"
            
            episode_details_map = self._setup_initial_filtered_dirs_and_copy_files(
                project_name, 
                source_conf, 
                project_input_base_dir,
                project_output_base_dir
            )
            
            per_source_processing_data[input_subpath_str] = {
                'name_for_log': source_display_name,
                'episode_details_map': episode_details_map, 
                'output_base_this_stage': current_stage_output_full_dir / input_subpath_str, 
                'target_speaker_label': speaker_label
            }

            for ep_data in episode_details_map.values():
                all_files_for_master_profile.extend(ep_data.get('initial_audio_files', []))
        
        if not all_files_for_master_profile:
            tqdm.write("ERROR: No audio files found across any sources after initial setup. Cannot create master profile. Aborting.")
            return

        master_centroid, master_embeddings_map = self._calculate_speaker_profile(
            all_files_for_master_profile,
            f"MASTER PROFILE for project '{project_name}' (all sources)"
        )

        if master_centroid is None:
            tqdm.write("ERROR: Could not calculate master speaker profile. Aborting outlier detection phase.")
            return
        tqdm.write("--- Master Profile Creation Complete ---")

        tqdm.write("\n--- Phase 2: Per-Source Outlier Detection (using Master Profile) ---")
        for source_input_subpath_key, source_data in per_source_processing_data.items():
            source_name_log = source_data['name_for_log']
            episode_details_this_source = source_data['episode_details_map']
            source_output_base_path = source_data['output_base_this_stage']

            tqdm.write(f"\n----- Processing Source: {source_name_log} -----")
            if not episode_details_this_source:
                tqdm.write(f"  No processable episode data for source '{source_name_log}'. Skipping outlier detection.")
                continue
        
            source_total_definite_moved, source_total_uncertain_moved = 0, 0
            source_all_episode_reports: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: {"definite": [], "uncertain": []})

            files_in_this_source_with_master_embeddings = []
            for ep_name, ep_dat in episode_details_this_source.items():
                for audio_f_path in ep_dat.get('initial_audio_files',[]):
                    if audio_f_path in master_embeddings_map:
                        files_in_this_source_with_master_embeddings.append(
                            {'path': audio_f_path, 'embedding': master_embeddings_map[audio_f_path]}
                        )
            
            if not files_in_this_source_with_master_embeddings:
                tqdm.write(f"  No audio files from source '{source_name_log}' have embeddings in the master map. Skipping its outlier detection.")
                continue

            embeddings_by_episode: Dict[str, List[Dict[str,Any]]] = defaultdict(list)
            for item_with_embed in files_in_this_source_with_master_embeddings:
                episode_name_from_path = item_with_embed['path'].parent.parent.name
                embeddings_by_episode[episode_name_from_path].append(item_with_embed)

            for ep_name, ep_audio_embeddings_list in tqdm(embeddings_by_episode.items(), desc=f"  Outlier check for episodes in {source_name_log}", leave=False):
                current_episode_base_dir = episode_details_this_source[ep_name]['speaker_dir_current_stage']
                
                reports, moved_d, moved_u = self._identify_and_move_outliers_for_episode(
                    ep_name, current_episode_base_dir, 
                    ep_audio_embeddings_list, master_centroid
                )
                source_total_definite_moved += moved_d
                source_total_uncertain_moved += moved_u
                source_all_episode_reports[ep_name] = reports
            
            self._print_source_summary(
                source_name_log, source_output_base_path,
                source_total_definite_moved, source_total_uncertain_moved,
                source_all_episode_reports, episode_details_this_source
            )
            self._check_episode_outlier_rates(
                 source_name_log, episode_details_this_source, source_all_episode_reports
            )
            tqdm.write(f"----- Finished Outlier Processing for Source: {source_name_log} -----")
        
        tqdm.write("\n" + "="*60)
        tqdm.write(f"OutlierDetector: All sources processed for project '{project_name}'.")
        tqdm.write(f"  Final outputs in: {project_output_base_dir.resolve()}")
        tqdm.write("="*60)

# For standalone testing:
if __name__ == '__main__':
    config_file = Path("config.yaml")
    if not config_file.exists():
        config_file = Path("../config.yaml") 
    
    if not config_file.exists():
        print(f"ERROR: {config_file.name} not found in current or parent directory for standalone test.")
        exit(1)

    # Load .env file from project root if it exists
    dotenv_path = config_file.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        # print(f"Loaded environment variables from: {dotenv_path}")

    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Could not load or parse {config_file.resolve()}: {e}")
        exit(1)

    project_root_for_test = config_file.parent.resolve()
    if Path.cwd() != project_root_for_test:
        print(f"Standalone test: Changing CWD to project root: {project_root_for_test}")
        os.chdir(project_root_for_test)

    project_setup = cfg.get("project_setup")
    if not project_setup:
        print("ERROR: 'project_setup' section not found in config.yaml.")
        exit(1)

    project_name_for_test = project_setup.get("project_name")
    if not project_name_for_test:
        print("ERROR: 'project_name' not found in project_setup config.")
        exit(1)
        
    output_base_dir_str = project_setup.get("output_base_dir")
    if not output_base_dir_str:
        print("ERROR: 'output_base_dir' not found in project_setup config.")
        exit(1)
    
    output_base_abs = Path(output_base_dir_str)
    if not output_base_abs.is_absolute():
        # All paths in config are relative to project root (config file location)
        output_base_abs = (project_root_for_test / output_base_abs).resolve()
    
    # This is the base processing directory for the specific project (e.g., /path/to/output_base_dir/ProjectName)
    processing_project_dir_abs = output_base_abs / project_name_for_test
    
    outlier_cfg = cfg.get("find_outliers_config")
    if not outlier_cfg:
        print("ERROR: 'find_outliers_config' section not found in config.yaml.")
        exit(1)
    
    project_audio_sources_cfg = outlier_cfg.get("project_audio_sources")
    input_stage_rel_folder = outlier_cfg.get("input_stage_folder_name")
    output_stage_rel_folder = outlier_cfg.get("output_stage_folder_name")

    if not project_audio_sources_cfg or not isinstance(project_audio_sources_cfg, list):
        print("ERROR: 'project_audio_sources' list not found or invalid in 'find_outliers_config'.")
        exit(1)
    if not input_stage_rel_folder:
        print("ERROR: 'input_stage_folder_name' not found in 'find_outliers_config'.")
        exit(1)
    if not output_stage_rel_folder:
        print("ERROR: 'output_stage_folder_name' not found in 'find_outliers_config'.")
        exit(1)
        
    for idx, src_cfg_item in enumerate(project_audio_sources_cfg):
        if not isinstance(src_cfg_item, dict) or \
           "input_subpath_relative_to_input_stage" not in src_cfg_item or \
           "target_speaker_label" not in src_cfg_item:
            print(f"ERROR: Invalid structure for project_audio_sources item at index {idx}. ")
            exit(1)

    # Derive absolute input and output paths for the standalone test
    # Input is: <output_base_dir>/<project_name>/<input_stage_folder_name>
    input_dir_for_standalone_test = processing_project_dir_abs / input_stage_rel_folder
    # Output is: <output_base_dir>/<project_name>/<output_stage_folder_name>
    output_dir_for_standalone_test = processing_project_dir_abs / output_stage_rel_folder

    print(f"--- Running OutlierDetector Standalone Test for project: {project_name_for_test} ---")
    print(f"  Input base from: {input_dir_for_standalone_test}")
    print(f"  Output base to: {output_dir_for_standalone_test}")

    try:
        # Pass the full cfg as global_config
        detector = OutlierDetector(global_config=cfg, outlier_config=outlier_cfg)
        detector.run_outlier_identification_for_project(
            project_name=project_name_for_test,
            project_audio_sources_config=project_audio_sources_cfg,
            project_input_base_dir=input_dir_for_standalone_test,
            project_output_base_dir=output_dir_for_standalone_test
        )
    except FileNotFoundError as e_fnf:
        print(f"FATAL FILE NOT FOUND ERROR during OutlierDetector operation: {e_fnf}")
    except RuntimeError as e_rt:
        print(f"FATAL RUNTIME ERROR during OutlierDetector operation: {e_rt}")
    except Exception as e_main:
        print(f"UNEXPECTED ERROR during OutlierDetector standalone test: {type(e_main).__name__} - {e_main}")
        traceback.print_exc()
    
    # final_output_location is now output_dir_for_standalone_test
    print(f"--- Standalone Test Finished. Check output in: {output_dir_for_standalone_test.resolve()} ---")