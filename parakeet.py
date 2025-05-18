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
import yaml # For loading config in __main__
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv # For consistency


class ParakeetASRProcessor:
    """Processes audio files using a Parakeet ASR model for transcription and flagging."""

    def __init__(self, global_config: Dict[str, Any], parakeet_config: Dict[str, Any]):
        """
        Initializes the ParakeetASRProcessor with settings from configuration.
        Args:
            global_config: Contains global project settings. CWD is assumed to be project root.
            parakeet_config: Contains settings specific to Parakeet ASR processing.
        """
        # self.project_root_dir = Path(global_config["project_settings"]["project_root_dir"]) # REMOVED - CWD is project root
        
        # ASR model and processing parameters
        self.asr_model_name: str = parakeet_config["asr_model_name"]
        self.batch_size: int = parakeet_config["batch_size"]
        self.device: str = parakeet_config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Timestamp and flagging parameters
        self.timestamp_deviation_threshold_sec: float = parakeet_config["timestamp_deviation_threshold_sec"]
        self.padding_sec: float = parakeet_config["padding_sec"]
        
        # Output subfolder names from config
        self.flagged_cropped_dir_name: str = parakeet_config["output_subfolder_names"]["flagged_cropped"]
        self.processed_main_dir_name: str = parakeet_config["output_subfolder_names"]["processed_main"]
        self.audio_file_format_glob: str = parakeet_config.get("audio_file_format_glob", "*.wav")

        self.asr_model: Optional[nemo_asr.models.ASRModel] = None
        self._load_asr_model()
        tqdm.write("ParakeetASRProcessor initialized.")

    def _load_asr_model(self):
        tqdm.write(f"ParakeetASR: Loading ASR model: {self.asr_model_name} on {self.device}...")
        try:
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.asr_model_name, 
                map_location=self.device
            )
            self.asr_model.eval()
            tqdm.write("ParakeetASR: ASR model loaded successfully.")
        except Exception as e:
            # Raising an error here as ASR model is critical
            raise RuntimeError(f"Fatal: Error loading ASR model '{self.asr_model_name}': {e}")

    def _get_asr_boundary_times(self, asr_hyp_result: Any, audio_duration_seconds: float) -> Tuple[float, float]:
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
                except (ValueError, TypeError, KeyError): 
                    pass # Keep default full duration if error
        return asr_start_sec, asr_end_sec

    def run_asr_for_project(
        self, 
        project_input_audio_dir: Path, 
        project_output_base_dir: Path
    ):
        if not self.asr_model:
            tqdm.write("ParakeetASR ERROR: ASR model not loaded. Aborting.")
            return

        flagged_cropped_output_path = project_output_base_dir / self.flagged_cropped_dir_name
        processed_main_output_path = project_output_base_dir / self.processed_main_dir_name

        for p in [flagged_cropped_output_path, processed_main_output_path]:
            if p.exists(): 
                tqdm.write(f"ParakeetASR: Cleaning existing output directory: {p}")
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
        
        tqdm.write(f"ParakeetASR: Input audio from: {project_input_audio_dir.resolve()}")
        tqdm.write(f"ParakeetASR: Main output to: {processed_main_output_path.resolve()}")
        tqdm.write(f"ParakeetASR: Flagged output to: {flagged_cropped_output_path.resolve()}")

        audio_files = sorted(list(project_input_audio_dir.rglob(self.audio_file_format_glob)))
        if not audio_files: 
            tqdm.write(f"ParakeetASR: No '{self.audio_file_format_glob}' files found in {project_input_audio_dir}. Aborting."); 
            return
        tqdm.write(f"ParakeetASR: Found {len(audio_files)} '{self.audio_file_format_glob}' files.")

        flagged_count = 0
        processed_main_count = 0

        for i in tqdm(range(0, len(audio_files), self.batch_size), desc="ASR Processing Batches", unit="batch"):
            batch_paths_str = [str(p) for p in audio_files[i:i + self.batch_size]]
            batch_paths_obj = audio_files[i:i + self.batch_size]

            try:
                asr_results = self.asr_model.transcribe(
                    batch_paths_str,
                    batch_size=len(batch_paths_str),
                    timestamps=True,
                    verbose=False
                )
            except Exception as e:
                tqdm.write(f"  Error during ASR transcription for a batch: {e}. Skipping batch.")
                # Optional: log detailed errors to a file in the project root (CWD)
                debug_log_path = Path.cwd() / "debug_asr_errors.log"
                # Check if the path to the debug log file exists (or if the key exists in config to enable it)
                # For simplicity, we just check if the file exists or can be created by trying to open.
                # A more robust way would be a config flag.
                try:
                    with open(debug_log_path, "a") as log_f:
                        log_f.write(f"Batch starting with {batch_paths_str[0] if batch_paths_str else 'N/A'}: {e}\\n{traceback.format_exc()}\\n")
                except IOError:
                    tqdm.write(f"  Warning: Could not write to debug log at {debug_log_path}")
                continue

            if not asr_results or len(asr_results) != len(batch_paths_obj):
                tqdm.write(f"  Warning: Mismatch or no ASR results for batch. Expected {len(batch_paths_obj)}, got {len(asr_results) if asr_results else 0}. Skipping batch.")
                continue
            
            # Ensure asr_results is a list, even if a single string (text) is returned for a file
            if isinstance(asr_results, tuple) and len(asr_results) == 1 and isinstance(asr_results[0], list): # NeMo can return ([hypotheses], [beam_hypotheses])
                 asr_hypotheses = asr_results[0]
            elif isinstance(asr_results, list):
                 asr_hypotheses = asr_results
            else:
                tqdm.write(f"  Warning: Unexpected ASR result format for batch. Skipping. Format: {type(asr_results)}")
                continue
            
            if len(asr_hypotheses) != len(batch_paths_obj):
                tqdm.write(f"  Warning: Mismatch in ASR hypotheses count. Expected {len(batch_paths_obj)}, got {len(asr_hypotheses)}. Skipping batch.")
                continue


            for original_audio_path, asr_hyp_obj in zip(batch_paths_obj, asr_hypotheses):
                try:
                    full_audio_data, original_sr = sf.read(str(original_audio_path), dtype='float32')
                    if full_audio_data.ndim > 1: # Ensure mono for duration calculation and processing
                        full_audio_data = np.mean(full_audio_data, axis=1)
                    original_duration = len(full_audio_data) / original_sr if original_sr > 0 else 0.0
                    
                    # Handle cases where asr_hyp_obj might be a simple string (no timestamps) or an object
                    asr_text_to_save = ""
                    raw_asr_start_sec, raw_asr_end_sec = 0.0, original_duration

                    if isinstance(asr_hyp_obj, str): # Simple text result
                        asr_text_to_save = asr_hyp_obj
                    elif hasattr(asr_hyp_obj, 'text'): # NeMo ASR Hypothesis object
                        asr_text_to_save = asr_hyp_obj.text or ""
                        raw_asr_start_sec, raw_asr_end_sec = self._get_asr_boundary_times(asr_hyp_obj, original_duration)
                    else:
                        tqdm.write(f"    Warning: Unexpected ASR hypothesis type for {original_audio_path.name}: {type(asr_hyp_obj)}. Using empty text.")

                    # --- Step 1: Always copy original audio and ASR text to the main processed directory ---
                    main_output_audio_path = processed_main_output_path / original_audio_path.name
                    shutil.copy2(str(original_audio_path), main_output_audio_path)
                    
                    main_output_txt_path = processed_main_output_path / original_audio_path.with_suffix(".txt").name
                    with open(main_output_txt_path, 'w', encoding='utf-8') as f:
                        f.write(asr_text_to_save)
                    processed_main_count +=1

                    # --- Step 2: Check for flagging and if flagged, also save cropped version ---
                    start_deviation = abs(raw_asr_start_sec - 0.0)
                    end_deviation = abs(raw_asr_end_sec - original_duration)
                    
                    is_flagged = start_deviation > self.timestamp_deviation_threshold_sec or \
                                 end_deviation > self.timestamp_deviation_threshold_sec
                    
                    if is_flagged:
                        padded_asr_start_sec = raw_asr_start_sec - self.padding_sec
                        padded_asr_end_sec = raw_asr_end_sec + self.padding_sec

                        final_crop_start_sec = max(0.0, padded_asr_start_sec)
                        final_crop_end_sec = min(original_duration, padded_asr_end_sec)
                        
                        if final_crop_end_sec <= final_crop_start_sec: # Reset to raw if padding caused issues
                            if raw_asr_end_sec > raw_asr_start_sec:
                                final_crop_start_sec, final_crop_end_sec = raw_asr_start_sec, raw_asr_end_sec
                            else: # If raw also problematic, use full audio
                                final_crop_start_sec, final_crop_end_sec = 0.0, original_duration
                        
                        start_frame = int(final_crop_start_sec * original_sr)
                        end_frame = int(final_crop_end_sec * original_sr)

                        audio_to_save_in_flagged = full_audio_data
                        if original_duration > 0 and end_frame > start_frame and start_frame >= 0 and end_frame <= len(full_audio_data):
                            audio_to_save_in_flagged = full_audio_data[start_frame:end_frame]
                        
                        if audio_to_save_in_flagged.size == 0:
                            if full_audio_data.size > 0: audio_to_save_in_flagged = full_audio_data
                            else: continue # Skip flagged save if all audio is empty
                        
                        flagged_output_audio_path = flagged_cropped_output_path / original_audio_path.name
                        sf.write(str(flagged_output_audio_path), audio_to_save_in_flagged, original_sr, subtype='PCM_24') # Assuming wav output from find_outliers is PCM_24
                        
                        flagged_output_txt_path = flagged_cropped_output_path / original_audio_path.with_suffix(".txt").name
                        with open(flagged_output_txt_path, 'w', encoding='utf-8') as f:
                            f.write(asr_text_to_save)
                        flagged_count += 1
                
                except Exception as e_file:
                    tqdm.write(f"    Error processing file {original_audio_path.name}: {e_file}")
                    # traceback.print_exc() # Uncomment for detailed file error debugging

        tqdm.write("\n--- ParakeetASR Processing Complete ---")
        tqdm.write(f"  Total input files: {len(audio_files)}")
        tqdm.write(f"  Files processed and saved to '{self.processed_main_dir_name}': {processed_main_count}")
        tqdm.write(f"  Flagged files (cropped version saved to '{self.flagged_cropped_dir_name}'): {flagged_count}")
        tqdm.write(f"  Output base directory: {project_output_base_dir.resolve()}")

if __name__ == "__main__":
    config_file = Path("config.yaml")
    if not config_file.exists():
        config_file = Path("../config.yaml")
    
    if not config_file.exists():
        print(f"ERROR: {config_file.name} not found for standalone test.")
        exit(1)

    dotenv_path = config_file.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)

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
    output_base_dir_str = project_setup.get("output_base_dir")
    if not project_name_for_test or not output_base_dir_str:
        print("ERROR: 'project_name' or 'output_base_dir' not found in project_setup.")
        exit(1)

    output_base_abs = Path(output_base_dir_str)
    if not output_base_abs.is_absolute():
        output_base_abs = (project_root_for_test / output_base_abs).resolve()
    
    processing_project_dir_abs = output_base_abs / project_name_for_test

    parakeet_cfg = cfg.get("parakeet_asr_config")
    if not parakeet_cfg:
        print("ERROR: 'parakeet_asr_config' not found in config.yaml.")
        exit(1)

    # Determine input directory for Parakeet (output of find_outliers stage)
    find_outliers_cfg = cfg.get("find_outliers_config")
    if not find_outliers_cfg:
        print("ERROR: 'find_outliers_config' not found in config.yaml (needed to determine Parakeet input).")
        exit(1)
    
    fo_output_stage_rel_folder = find_outliers_cfg.get("output_stage_folder_name")
    if not fo_output_stage_rel_folder:
        print("ERROR: 'output_stage_folder_name' not in 'find_outliers_config'.")
        exit(1)

    # Parakeet input is typically from a specific speaker's 'filtered' subfolder from the outlier stage.
    # For standalone, we'll use the first source defined in find_outliers_config as an example.
    fo_project_sources = find_outliers_cfg.get("project_audio_sources")
    if not fo_project_sources or not isinstance(fo_project_sources, list) or not fo_project_sources[0]:
        print("ERROR: 'project_audio_sources' in 'find_outliers_config' is missing or invalid for determining Parakeet input.")
        exit(1)
    
    first_source_subpath_in_fo = Path(fo_project_sources[0]["input_subpath_relative_to_input_stage"])
    # Input for Parakeet: .../ProjectName/OutlierStageOutputFolder/SourceSubpath/filtered/
    parakeet_input_dir = processing_project_dir_abs / fo_output_stage_rel_folder / first_source_subpath_in_fo / "filtered"

    # Determine output directory for Parakeet
    parakeet_output_stage_rel_folder = parakeet_cfg.get("output_stage_folder_name")
    if not parakeet_output_stage_rel_folder:
        print("ERROR: 'output_stage_folder_name' not found in 'parakeet_asr_config'.")
        exit(1)
    
    # Parakeet output base: .../ProjectName/ParakeetStageOutputFolder/SourceSubpath/
    # The ParakeetASRProcessor itself will create 'flagged_cropped' and 'processed_main' inside this base.
    parakeet_output_base_dir_for_run = processing_project_dir_abs / parakeet_output_stage_rel_folder / first_source_subpath_in_fo

    if not parakeet_input_dir.is_dir():
        print(f"ERROR: Standalone test input audio directory for Parakeet not found: \n  {parakeet_input_dir.resolve()}")
        print(f"  Expected structure: .../{fo_output_stage_rel_folder}/{first_source_subpath_in_fo}/filtered/")
        exit(1)
            
    print(f"--- ParakeetASRProcessor Standalone Test ({project_name_for_test}) ---")
    print(f"  Input from : {parakeet_input_dir.relative_to(project_root_for_test) if parakeet_input_dir.is_relative_to(project_root_for_test) else parakeet_input_dir}")
    print(f"  Output base to: {parakeet_output_base_dir_for_run.relative_to(project_root_for_test) if parakeet_output_base_dir_for_run.is_relative_to(project_root_for_test) else parakeet_output_base_dir_for_run}")

    try:
        processor = ParakeetASRProcessor(global_config=cfg, parakeet_config=parakeet_cfg)
        processor.run_asr_for_project(
            project_input_audio_dir=parakeet_input_dir,
            project_output_base_dir=parakeet_output_base_dir_for_run 
        )
    except Exception as e_main: 
        print(f"ERROR during standalone Parakeet ASR test: {type(e_main).__name__} - {e_main}")
        traceback.print_exc()
    
    print(f"--- Test Finished. Output in: {parakeet_output_base_dir_for_run.resolve()} ---")