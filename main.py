#!/usr/bin/env python3
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

# Ensure the script directory is in the Python path for module imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_audio import AudioExtractor
from vocal_removal import VocalRemover
from normalize import AudioNormalizer
from llm_call import LLMDiarizer
from align import AudioTranscriptAligner
from find_outliers import OutlierDetector
from parakeet import ParakeetASRProcessor

# Define the order of pipeline stages
PIPELINE_STAGES = [
    "extract_audio",
    "vocal_removal",
    "normalize_audio",
    "llm_diarizer",
    "ctc_aligner",
    "find_outliers",
    "parakeet_asr"
]

class PipelineOrchestrator:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_file_path = Path(config_path).resolve()
        self.project_root_dir = self.config_file_path.parent

        try:
            with open(self.config_file_path, 'r') as f:
                self.config: Dict[str, Any] = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"ERROR: Configuration file '{self.config_file_path}' not found.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"ERROR: Could not parse configuration file '{self.config_file_path}': {e}")
            sys.exit(1)

        self.project_setup = self.config.get("project_setup", {})
        if not self.project_setup:
            print("ERROR: 'project_setup' section is missing in config.yaml.")
            sys.exit(1)

        self.project_name = self.project_setup.get("project_name")
        self.video_source_parent_dir = Path(self.project_setup.get("video_source_parent_dir", "./video_sources"))
        self.output_base_dir = Path(self.project_setup.get("output_base_dir", "./processing_output"))

        if not self.project_name:
            print("ERROR: 'project_name' is missing in project_setup in config.yaml.")
            sys.exit(1)

        # Resolve paths relative to project_root_dir if they are not absolute
        if not self.video_source_parent_dir.is_absolute():
            self.video_source_parent_dir = (self.project_root_dir / self.video_source_parent_dir).resolve()
        if not self.output_base_dir.is_absolute():
            self.output_base_dir = (self.project_root_dir / self.output_base_dir).resolve()
            
        self.processing_output_project_dir = (self.output_base_dir / self.project_name).resolve()

        self.pipeline_control_settings = self.config.get("pipeline_control", {})

        if Path.cwd() != self.project_root_dir:
            print(f"Changing CWD to project root: {self.project_root_dir}")
            os.chdir(self.project_root_dir)
        
        self.stage_runners: Dict[str, Callable[[], None]] = {
            PIPELINE_STAGES[0]: self._run_extract_audio,
            PIPELINE_STAGES[1]: self._run_vocal_removal,
            PIPELINE_STAGES[2]: self._run_normalize_audio,
            PIPELINE_STAGES[3]: self._run_llm_diarizer,
            PIPELINE_STAGES[4]: self._run_ctc_aligner,
            PIPELINE_STAGES[5]: self._run_find_outliers,
            PIPELINE_STAGES[6]: self._run_parakeet_asr,
        }

    def _get_stage_config(self, stage_key_in_pipeline_list: str) -> Dict[str, Any]:
        # Stage config blocks are now consistently named e.g., "extract_audio_config"
        config_block_name = f"{stage_key_in_pipeline_list}_config"
        stage_config = self.config.get(config_block_name)
        if not stage_config:
            print(f"Warning: Configuration for stage '{stage_key_in_pipeline_list}' (expected block: '{config_block_name}') not found. Skipping stage if it runs.")
            return {}
        return stage_config

    def _get_stage_io_paths(self,
                            current_stage_key: str,
                            prev_stage_key: Optional[str] = None,
                            input_subfolder_from_prev_stage: Optional[str] = None,
                            output_subfolder_for_current_stage: Optional[str] = None
                           ) -> Tuple[Optional[Path], Path]:
        
        current_stage_cfg = self._get_stage_config(current_stage_key)
        current_stage_relative_output_folder = current_stage_cfg.get("output_stage_folder_name")
        if not current_stage_relative_output_folder:
            # Fallback if somehow missing, though schema implies it should be there
            current_stage_relative_output_folder = f"{current_stage_key}_output_MISSING_IN_CONFIG"
            print(f"Warning: 'output_stage_folder_name' not found for stage '{current_stage_key}'. Using default: {current_stage_relative_output_folder}")

        # Output directory for the current stage
        output_dir = self.processing_output_project_dir / current_stage_relative_output_folder
        if output_subfolder_for_current_stage:
            output_dir = output_dir / output_subfolder_for_current_stage
        
        input_dir: Optional[Path] = None
        if prev_stage_key:
            prev_stage_cfg = self._get_stage_config(prev_stage_key)
            prev_stage_relative_output_folder = prev_stage_cfg.get("output_stage_folder_name")
            if prev_stage_relative_output_folder:
                input_dir = self.processing_output_project_dir / prev_stage_relative_output_folder
                if input_subfolder_from_prev_stage:
                    input_dir = input_dir / input_subfolder_from_prev_stage
            else:
                print(f"Warning: Could not determine input directory for '{current_stage_key}' because 'output_stage_folder_name' is missing for previous stage '{prev_stage_key}'.")
        
        return input_dir, output_dir

    def _run_extract_audio(self):
        stage_key = PIPELINE_STAGES[0]
        print(f"\n--- Running {stage_key} ---")
        cfg = self._get_stage_config(stage_key)
        if not cfg: return

        # Video source is now {video_source_parent_dir}/{project_name}
        video_source_project_dir = self.video_source_parent_dir / self.project_name
        _, output_dir_base = self._get_stage_io_paths(stage_key)
        # AudioExtractor expects the base output dir, it creates 'orig' internally

        extractor = AudioExtractor(config=cfg) # Pass only stage-specific config
        # The 'config' passed to AudioExtractor is extract_audio_config.
        # It does not need global_config or project_root_dir as it's self-contained for paths.
        extractor.run_extraction_for_project(video_source_project_dir, output_dir_base)
        print(f"--- {stage_key} Finished ---")

    def _run_vocal_removal(self):
        stage_key = PIPELINE_STAGES[1]
        print(f"\n--- Running {stage_key} ---")
        cfg = self._get_stage_config(stage_key)
        if not cfg: return
        
        input_dir, output_dir_base = self._get_stage_io_paths(
            current_stage_key=stage_key,
            prev_stage_key=PIPELINE_STAGES[0],
            input_subfolder_from_prev_stage="orig" # Extracted audio is in an 'orig' subfolder
        )
        if not input_dir: 
            print(f"Error: Could not determine input for {stage_key}. Input directory is None."); return

        # VocalRemover needs global_config for model paths and vocal_remover_specific_config
        remover = VocalRemover(global_config=self.config, vocal_remover_specific_config=cfg)
        remover.run_separation_for_project(input_dir, output_dir_base) # It creates 'vocals' subfolder
        print(f"--- {stage_key} Finished ---")

    def _run_normalize_audio(self):
        stage_key = PIPELINE_STAGES[2]
        print(f"\n--- Running {stage_key} ---")
        cfg = self._get_stage_config(stage_key)
        if not cfg: return
        
        input_dir, output_dir = self._get_stage_io_paths(
            current_stage_key=stage_key,
            prev_stage_key=PIPELINE_STAGES[1],
            input_subfolder_from_prev_stage="vocals" # Vocal remover outputs to 'vocals' subfolder
        )
        if not input_dir: 
            print(f"Error: Could not determine input for {stage_key}. Input directory is None."); return
        
        # AudioNormalizer takes its own config block
        normalizer = AudioNormalizer(config=cfg)
        normalizer.run_normalization_for_project(input_dir, output_dir)
        print(f"--- {stage_key} Finished ---")

    def _run_llm_diarizer(self):
        stage_key = PIPELINE_STAGES[3]
        print(f"\n--- Running {stage_key} ---")
        cfg = self._get_stage_config(stage_key)
        if not cfg: return
        
        # Input for LLM is the normalized audio
        normalized_audio_input_dir, output_json_dir = self._get_stage_io_paths(
            current_stage_key=stage_key,
            prev_stage_key=PIPELINE_STAGES[2] # Input from normalize_audio
        )
        if not normalized_audio_input_dir: 
            print(f"Error: Could not determine normalized audio input for {stage_key}. Input directory is None."); return

        # LLM Diarizer also needs access to the *original* extracted audio for full context if enabled by its config.
        # The original extracted audio is output by PIPELINE_STAGES[0] ('extract_audio') into its 'orig' subfolder.
        extract_audio_cfg = self._get_stage_config(PIPELINE_STAGES[0])
        extract_audio_stage_folder = extract_audio_cfg.get("output_stage_folder_name")
        if not extract_audio_stage_folder:
            print(f"Error: Could not determine output folder for '{PIPELINE_STAGES[0]}' needed by LLM Diarizer for original audio. Skipping original audio part.");
            original_extracted_audio_dir = None # Or handle error more strictly
        else:
            original_extracted_audio_dir = self.processing_output_project_dir / extract_audio_stage_folder / "orig"

        # LLMDiarizer needs global_config for project_setup (characters) and its own diarizer_config
        diarizer = LLMDiarizer(global_config=self.config, diarizer_config=cfg)
        diarizer.run_diarization_for_project(
            project_normalized_audio_dir=normalized_audio_input_dir,
            project_original_extracted_audio_dir=original_extracted_audio_dir, # Can be None
            project_output_diarized_json_dir=output_json_dir
        )
        print(f"--- {stage_key} Finished ---")

    def _run_ctc_aligner(self):
        stage_key = PIPELINE_STAGES[4]
        print(f"\n--- Running {stage_key} ---")
        cfg = self._get_stage_config(stage_key)
        if not cfg: return

        # Input JSONs from LLM Diarizer
        llm_json_input_dir, _ = self._get_stage_io_paths( # Output path of aligner is not directly used here by _get_stage_io_paths
            current_stage_key=stage_key, # current_stage_key is aligner for its config
            prev_stage_key=PIPELINE_STAGES[3]  # prev_stage_key is llm_diarizer for its output
        )
        if not llm_json_input_dir:
            print(f"Error: Could not determine LLM JSON input for {stage_key}. Input directory is None."); return

        # Input Normalized Audio from Audio Normalizer
        norm_stage_cfg = self._get_stage_config(PIPELINE_STAGES[2])
        norm_output_folder = norm_stage_cfg.get("output_stage_folder_name")
        if not norm_output_folder:
            print(f"Error: Could not determine output folder for '{PIPELINE_STAGES[2]}' needed by CTC Aligner."); return
        normalized_audio_input_dir = self.processing_output_project_dir / norm_output_folder
        
        # Aligner's own output folder name is in its config
        aligner_output_stage_folder_name = cfg.get("output_stage_folder_name")
        if not aligner_output_stage_folder_name:
             print(f"Error: 'output_stage_folder_name' missing in {stage_key} config."); return


        # AudioTranscriptAligner needs global_config for model paths and its own aligner_config
        aligner = AudioTranscriptAligner(global_config=self.config, aligner_config=cfg)
        aligner.run_alignment_for_project(
            project_llm_json_input_dir=llm_json_input_dir,
            project_normalized_audio_input_dir=normalized_audio_input_dir,
            project_name=self.project_name, # Pass project_name explicitly
            # project_output_parent_dir is where the {project_name}/{stage_folder} structure lives
            project_output_parent_dir=self.processing_output_project_dir.parent, # Pass the parent of project specific output dir
            stage_output_folder_name=aligner_output_stage_folder_name # Relative name for aligner's output
        )
        print(f"--- {stage_key} Finished ---")

    def _run_find_outliers(self):
        stage_key = PIPELINE_STAGES[5]
        print(f"\n--- Running {stage_key} ---")
        cfg = self._get_stage_config(stage_key)
        if not cfg: return

        # `input_stage_folder_name` from find_outliers_config tells us which previous stage's output to use
        # This is ALREADY a relative path name like "05_aligned_clips"
        input_stage_folder_name_for_outliers = cfg.get("input_stage_folder_name")
        if not input_stage_folder_name_for_outliers:
            print(f"Error: 'input_stage_folder_name' not defined in config for '{stage_key}'. Skipping."); return

        # `output_stage_folder_name` from find_outliers_config is for its own output
        output_stage_folder_name_for_outliers = cfg.get("output_stage_folder_name")
        if not output_stage_folder_name_for_outliers:
            print(f"Error: 'output_stage_folder_name' not defined in config for '{stage_key}'. Skipping."); return
            
        project_audio_sources_cfg = cfg.get("project_audio_sources")
        if not project_audio_sources_cfg:
            print(f"Error: 'project_audio_sources' not defined in config for '{stage_key}'. Skipping.")
            return

        # OutlierDetector needs global_config for models and its own outlier_config
        detector = OutlierDetector(global_config=self.config, outlier_config=cfg)
        # run_outlier_identification_for_project expects:
        # - project_name
        # - project_audio_sources_config (list from its own config section)
        # - input_stage_folder_name (relative name of the previous stage's output folder, e.g., "05_aligned_clips")
        # - output_stage_folder_name (relative name for its own output folder, e.g., "06_outliers_filtered")
        detector.run_outlier_identification_for_project(
            project_name=self.project_name,
            project_audio_sources_config=project_audio_sources_cfg,
            input_stage_folder_name=input_stage_folder_name_for_outliers,
            output_stage_folder_name=output_stage_folder_name_for_outliers
            # Note: OutlierDetector internally uses global_config.processing_output_parent_dir
            # which is now self.processing_output_project_dir.parent
        )
        print(f"--- {stage_key} Finished ---")

    def _run_parakeet_asr(self):
        stage_key = PIPELINE_STAGES[6]
        print(f"\n--- Running {stage_key} ---")
        parakeet_cfg = self._get_stage_config(stage_key)
        if not parakeet_cfg: return

        # Parakeet's input comes from find_outliers' output.
        # We need find_outliers_config to know its output_stage_folder_name and project_audio_sources.
        find_outliers_stage_cfg = self._get_stage_config(PIPELINE_STAGES[5])
        find_outliers_output_folder_name = find_outliers_stage_cfg.get("output_stage_folder_name")
        if not find_outliers_output_folder_name:
            print(f"Error: Could not determine output folder for '{PIPELINE_STAGES[5]}' needed by {stage_key}."); return
        
        project_audio_sources_from_outliers_cfg = find_outliers_stage_cfg.get("project_audio_sources", [])
        if not project_audio_sources_from_outliers_cfg:
            print(f"Warning: 'project_audio_sources' not found in '{PIPELINE_STAGES[5]}' config. {stage_key} might not find inputs if it relies on this structure.")

        # Parakeet's own output folder name
        parakeet_output_stage_folder_name = parakeet_cfg.get("output_stage_folder_name")
        if not parakeet_output_stage_folder_name:
            print(f"Error: 'output_stage_folder_name' missing in {stage_key} config."); return

        # ParakeetASRProcessor needs global_config for model paths and its own parakeet_config
        processor = ParakeetASRProcessor(global_config=self.config, parakeet_config=parakeet_cfg)

        # Iterate through each source defined in find_outliers_config, as Parakeet processes them one by one.
        for source_config_item in project_audio_sources_from_outliers_cfg:
            # This is the subpath *within the aligner's output* that find_outliers processed.
            # e.g., "Momo_Ayase_aligned"
            source_subpath_from_aligner_output = Path(source_config_item["input_subpath_relative_to_input_stage"])
            
            # Parakeet's input for this source is:
            # {output_base_dir}/{project_name}/{find_outliers_output_folder_name}/{source_subpath_from_aligner_output}/filtered/
            parakeet_input_dir_for_this_source = self.processing_output_project_dir / find_outliers_output_folder_name / source_subpath_from_aligner_output / "filtered"
            
            # Parakeet's output base for this source is:
            # {output_base_dir}/{project_name}/{parakeet_output_stage_folder_name}/{source_subpath_from_aligner_output}/
            # ParakeetASRProcessor will create its "flagged_cropped" and "processed_main" subdirs inside this.
            parakeet_output_base_for_this_source = self.processing_output_project_dir / parakeet_output_stage_folder_name / source_subpath_from_aligner_output

            if not parakeet_input_dir_for_this_source.is_dir():
                print(f"  Warning: Input directory for {stage_key} not found for source '{source_subpath_from_aligner_output}':")
                print(f"    {parakeet_input_dir_for_this_source}. Skipping this source.")
                continue
            
            print(f"  Processing {stage_key} for source: {source_subpath_from_aligner_output}")
            processor.run_asr_for_project(
                project_input_audio_dir=parakeet_input_dir_for_this_source,
                project_output_base_dir=parakeet_output_base_for_this_source # Parakeet creates subdirs like 'flagged_cropped' here
            )
        print(f"--- {stage_key} Finished ---")

    def run_pipeline(self):
        run_only_stage = self.pipeline_control_settings.get("run_only_stage")
        start_stage = self.pipeline_control_settings.get("start_stage")
        end_stage = self.pipeline_control_settings.get("end_stage")

        if run_only_stage:
            if run_only_stage not in self.stage_runners:
                print(f"ERROR: Specified stage '{run_only_stage}' in config is not valid. Available: {PIPELINE_STAGES}")
                return
            print(f"Running ONLY stage (from config): {run_only_stage}")
            self.stage_runners[run_only_stage]()
            print(f"\nPipeline run finished for stage: {run_only_stage}.")
            return

        try:
            start_index = PIPELINE_STAGES.index(start_stage) if start_stage else 0
            end_index = PIPELINE_STAGES.index(end_stage) if end_stage else len(PIPELINE_STAGES) - 1
        except ValueError as e:
            print(f"ERROR: Invalid start or end stage specified in config: {e}. Available: {PIPELINE_STAGES}")
            return

        if start_index > end_index:
            print("ERROR: Start stage in config cannot be after end stage.")
            return

        active_stages = PIPELINE_STAGES[start_index : end_index + 1]
        print(f"Running pipeline from stage '{active_stages[0]}' to '{active_stages[-1]}' (from config).")
        for stage_name in active_stages:
            if stage_name in self.stage_runners:
                self.stage_runners[stage_name]()
            else:
                print(f"Warning: Stage '{stage_name}' is defined in PIPELINE_STAGES but has no runner method. Skipping.")
        
        print("\nPipeline execution complete.")

def main():
    # Allow overriding config path via environment variable, otherwise default to "config.yaml" in CWD
    # The orchestrator now resolves this path and sets CWD to its parent.
    config_file_param = os.getenv("TTSIZER_CONFIG_PATH", "config.yaml")
    orchestrator = PipelineOrchestrator(config_path=config_file_param)
    orchestrator.run_pipeline()

if __name__ == "__main__":
    main()
