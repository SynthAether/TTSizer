import warnings
warnings.filterwarnings("ignore")

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm.auto import tqdm
import yaml # For loading config in __main__
from dotenv import load_dotenv
import google.generativeai as genai # Removed try-except
from google.generativeai import types as genai_types # Removed try-except
import time

class LLMDiarizer: # Renamed class
    """Handles diarization and transcription using a large language model (LLM)."""
    def __init__(self, global_config: Dict[str, Any], diarizer_config: Dict[str, Any]):
        """
        Initializes the LLMDiarizer.
        Args:
            global_config: The full configuration dictionary, including 'project_setup' and 'global_model_paths'.
            diarizer_config: Configuration specific to this LLM diarization stage.
        """
        self.project_setup = global_config.get("project_setup", {})
        self.project_name = self.project_setup.get("project_name", "UnknownProject")
        self.target_speakers = self.project_setup.get("target_speaker_labels_for_diarization", [])
        self.primary_character = self.project_setup.get("primary_character_for_llm_prompt", "")

        self.model_name: str = diarizer_config["model_name"]
        self.temperature: float = diarizer_config["temperature"]
        self.top_p: float = diarizer_config["top_p"]
        self.prompt_template_file = Path(diarizer_config["prompt_template_file"]) # Path relative to project root
        self.skip_if_output_exists: bool = diarizer_config.get("skip_if_output_exists", True)
        
        # API key is expected to be set as an environment variable by the user/orchestrator environment
        # self.api_key = os.getenv(diarizer_config.get("api_key_env_var", "GEMINI_API_KEY"))
        # if not self.api_key:
        #     raise ValueError("LLM API key not found in environment variables.")
        # genai.configure(api_key=self.api_key)
        # Configuration of the API key is now done externally before calling this class.
        # For safety, we can check if it *has* been configured.
        # if not genai. υπάρχει_configured_api_key(): # This is a placeholder for a real check if available
        #     tqdm.write("Warning: Gemini API key may not be configured. Calls might fail.")
        # The above check was a placeholder and has been removed.
        # It's assumed genai is configured externally or picks up credentials from the environment.

        self.model = genai.GenerativeModel(self.model_name)
        self.prompt_template: str = ""
        if self.prompt_template_file.exists():
            with open(self.prompt_template_file, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
        else:
            raise FileNotFoundError(f"Prompt template file not found: {self.prompt_template_file.resolve()}")
        tqdm.write(f"LLMDiarizer initialized for project '{self.project_name}'. Model: {self.model_name}")

    def _generate_content_with_retry(self, model_input: List[Any], max_retries: int = 3, initial_delay: int = 5) -> Optional[str]:
        """Generates content from the LLM with retry logic for common API errors."""
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            try:
                response = self.model.generate_content(model_input)
                if response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
                else:
                    tqdm.write(f"    LLM response missing content or parts. Full response: {response}")
                    return None # Or raise an error after retries
            except Exception as e: # Catching a broad range of google.api_core.exceptions
                # Specific exceptions to retry on (customize as needed):
                # google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.ServiceUnavailable,
                # google.api_core.exceptions.DeadlineExceeded, google.api_core.exceptions.InternalServerError
                # A more general check for typical retryable HTTP error codes if possible, or error messages.
                # For now, this is a simplified retry for any exception from generate_content.
                retries += 1
                tqdm.write(f"    LLM API call failed (attempt {retries}/{max_retries}): {type(e).__name__} - {e}")
                if retries >= max_retries:
                    tqdm.write(f"    LLM API call failed after {max_retries} attempts. Giving up.")
                    return None # Or re-raise the last exception
                time.sleep(delay)
                delay *= 2 # Exponential backoff
        return None

    def _process_single_episode(self, normalized_episode_audio_path: Path, 
                                original_episode_audio_path: Optional[Path], 
                                output_json_path: Path):
        tqdm.write(f"  Processing episode: {normalized_episode_audio_path.name}")
        
        if self.skip_if_output_exists and output_json_path.exists():
            tqdm.write(f"    Output JSON already exists: {output_json_path.name}. Skipping.")
            return

        # Use normalized audio for the LLM API call
        audio_file_for_llm = genai.upload_file(path=str(normalized_episode_audio_path))
        tqdm.write(f"    Uploaded {normalized_episode_audio_path.name} to LLM service.")

        # Prepare the prompt
        # Ensure target_speakers is a comma-separated string for the prompt
        characters_str = ", ".join(self.target_speakers) if self.target_speakers else "Not Specified"
        # Replace placeholders in the prompt template
        # Using primary_character for {CHARACTER_1} as per old logic, can be refined
        # Fallback for primary_character if not in target_speakers or target_speakers is empty
        char1_for_prompt = self.primary_character
        if not char1_for_prompt or (self.target_speakers and char1_for_prompt not in self.target_speakers):
            char1_for_prompt = self.target_speakers[0] if self.target_speakers else "First Main Character"
        
        current_prompt = self.prompt_template.replace("{ANIME_TITLE}", self.project_name)
        current_prompt = current_prompt.replace("{CHARACTERS_OF_INTEREST}", characters_str)
        current_prompt = current_prompt.replace("{CHARACTER_1}", char1_for_prompt)
        # TODO: Add replacement for {CHARACTER_2} etc. if prompt template supports more specific character slots

        model_input = [current_prompt, audio_file_for_llm]
        
        tqdm.write(f"    Calling LLM for diarization of {normalized_episode_audio_path.name}...")
        llm_output_text = self._generate_content_with_retry(model_input)

        # Clean up uploaded file from LLM service *after* getting the response
        try:
            genai.delete_file(audio_file_for_llm.name)
            # tqdm.write(f"    Cleaned up uploaded file: {audio_file_for_llm.name}")
        except Exception as e_del:
            tqdm.write(f"    Warning: Failed to delete uploaded file {audio_file_for_llm.name}: {e_del}")

        if not llm_output_text:
            tqdm.write(f"    LLM returned no text for {normalized_episode_audio_path.name}. Skipping JSON output.")
            return

        # Post-process the LLM output (assuming it's a JSON string or similar)
        # Remove markdown code block fences if present
        cleaned_json_text = llm_output_text.strip()
        if cleaned_json_text.startswith("```json"):
            cleaned_json_text = cleaned_json_text[7:]
        if cleaned_json_text.endswith("```"):
            cleaned_json_text = cleaned_json_text[:-3]
        cleaned_json_text = cleaned_json_text.strip()

        try:
            # Validate if the output is valid JSON before saving
            json_data = json.loads(cleaned_json_text)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            tqdm.write(f"    Successfully saved diarization JSON to: {output_json_path.name}")
        except json.JSONDecodeError as e:
            tqdm.write(f"    ERROR: LLM output for {normalized_episode_audio_path.name} was not valid JSON: {e}")
            tqdm.write(f"    Problematic text (first 500 chars): {cleaned_json_text[:500]}")
            # Optionally save the raw text for debugging
            debug_path = output_json_path.with_suffix(".raw_llm_output.txt")
            with open(debug_path, 'w', encoding='utf-8') as f_debug:
                f_debug.write(llm_output_text)
            tqdm.write(f"    Raw LLM output saved to: {debug_path.name}")

    def run_diarization_for_project(self, 
                                    project_normalized_audio_dir: Path, 
                                    project_original_extracted_audio_dir: Optional[Path],
                                    project_output_diarized_json_dir: Path):
        tqdm.write(f"\\nLLMDiarizer: Starting diarization for project: {self.project_name}")
        tqdm.write(f"  Normalized audio input from: {project_normalized_audio_dir.resolve()}")
        if project_original_extracted_audio_dir:
            tqdm.write(f"  Original extracted audio (for context/reference) from: {project_original_extracted_audio_dir.resolve()}")
        tqdm.write(f"  Outputting JSON to: {project_output_diarized_json_dir.resolve()}")
        project_output_diarized_json_dir.mkdir(parents=True, exist_ok=True)

        # Assuming normalized audio files are directly in project_normalized_audio_dir (e.g., episode.flac)
        # And original audio files (if used) have corresponding names in project_original_extracted_audio_dir
        normalized_audio_files = sorted(list(project_normalized_audio_dir.glob("*.flac")))
        if not normalized_audio_files:
            normalized_audio_files = sorted(list(project_normalized_audio_dir.glob("*.wav")))
        
        if not normalized_audio_files:
            tqdm.write(f"No .flac or .wav audio files found in {project_normalized_audio_dir}. Cannot proceed.")
            return

        for norm_audio_path in tqdm(normalized_audio_files, desc="Processing Episodes for LLM Diarization", unit="episode"):
            output_json_file_path = project_output_diarized_json_dir / norm_audio_path.with_suffix(".json").name
            
            original_audio_path_for_episode = None
            if project_original_extracted_audio_dir:
                # Try to find corresponding original audio file (could be .flac or .wav, etc.)
                # This assumes original and normalized files share the same stem.
                potential_orig_flac = project_original_extracted_audio_dir / norm_audio_path.with_suffix(".flac").name
                potential_orig_wav = project_original_extracted_audio_dir / norm_audio_path.with_suffix(".wav").name
                if potential_orig_flac.exists():
                    original_audio_path_for_episode = potential_orig_flac
                elif potential_orig_wav.exists():
                    original_audio_path_for_episode = potential_orig_wav
                # Add more extensions if needed

            self._process_single_episode(norm_audio_path, original_audio_path_for_episode, output_json_file_path)
        
        tqdm.write(f"LLMDiarizer: Finished diarization for project {self.project_name}.")

# For standalone testing:
if __name__ == '__main__':
    config_file = Path("config.yaml")
    if not config_file.exists():
        config_file = Path("../config.yaml") # For running from a 'scripts' or similar subfolder
    
    if not config_file.exists():
        print(f"ERROR: {config_file.name} not found in current or parent directory for standalone test.")
        exit(1)

    # Load environment variables from .env file if present (e.g., for API keys)
    # Ensure .env is in the project root (same directory as config.yaml)
    dotenv_path = config_file.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded environment variables from: {dotenv_path}")
    else:
        # Attempt to load .env from CWD if it's different and exists there,
        # though standard practice is .env at project root.
        if Path(".env").exists() and Path.cwd() != config_file.parent:
             load_dotenv()
             print(f"Loaded environment variables from: {Path.cwd() / '.env'}")
        # else: # No .env found, rely on globally set env vars
            # print(f"Info: No .env file found at {dotenv_path} or {Path.cwd() / '.env'}. Relying on globally set environment variables.")


    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Could not load or parse {config_file.resolve()}: {e}")
        exit(1)

    # Set CWD to project root (where config.yaml is)
    project_root_for_test = config_file.parent.resolve()
    if Path.cwd() != project_root_for_test:
        print(f"Standalone test: Changing CWD to project root: {project_root_for_test}")
        os.chdir(project_root_for_test)

    project_setup = cfg.get("project_setup")
    if not project_setup:
        print("ERROR: 'project_setup' section not found in config.yaml.")
        exit(1)

    project_name_for_test = project_setup.get("project_name")
    output_base_abs = Path(project_setup.get("output_base_dir"))
    if not output_base_abs.is_absolute():
        output_base_abs = (project_root_for_test / output_base_abs).resolve()
    
    processing_project_dir_abs = output_base_abs / project_name_for_test

    # LLM Diarizer config
    llm_diarizer_cfg = cfg.get("llm_diarizer_config")
    if not llm_diarizer_cfg:
        print("ERROR: 'llm_diarizer_config' not found in config.yaml.")
        exit(1)

    # Input: Normalized audio (from normalize_audio_config.output_stage_folder_name)
    norm_audio_cfg = cfg.get("normalize_audio_config", {})
    norm_audio_output_folder = norm_audio_cfg.get("output_stage_folder_name", "03_vocals_normalized_test")
    input_normalized_audio_dir = processing_project_dir_abs / norm_audio_output_folder

    # Input: Original extracted audio (for context) (from extract_audio_config.output_stage_folder_name / 'orig')
    extract_audio_cfg = cfg.get("extract_audio_config", {})
    extract_audio_output_folder = extract_audio_cfg.get("output_stage_folder_name", "01_extracted_audio_test")
    input_original_audio_dir = processing_project_dir_abs / extract_audio_output_folder / "orig"

    # Output: LLM Diarized JSON (llm_diarizer_config.output_stage_folder_name)
    llm_output_folder = llm_diarizer_cfg.get("output_stage_folder_name", "04_llm_diarized_json_STANDALONE_TEST")
    output_json_dir = processing_project_dir_abs / llm_output_folder

    if not input_normalized_audio_dir.is_dir():
        print(f"ERROR: Test input normalized audio directory not found: {input_normalized_audio_dir}")
        exit(1)
    # Original audio dir is optional for the script, so don't error if it's not there for test
    if not input_original_audio_dir.is_dir():
        print(f"Warning: Test input original audio directory not found: {input_original_audio_dir}. Proceeding without it.")

    print("\nInitializing LLMDiarizer for standalone test...")
    # Initialize LLMDiarizer with the diarizer-specific config and the full global config
    try:
        diarizer = LLMDiarizer(global_config=cfg, diarizer_config=llm_diarizer_cfg)
    except Exception as e:
        print(f"ERROR: Failed to initialize LLMDiarizer: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"Input normalized audio directory: {input_normalized_audio_dir}")
    print(f"Input original extracted audio directory: {input_original_audio_dir if input_original_audio_dir.is_dir() else 'Not found/used'}")
    print(f"Output JSON directory: {output_json_dir}")

    # Run the diarization process
    try:
        diarizer.run_diarization_for_project(
            project_normalized_audio_dir=input_normalized_audio_dir,
            project_original_extracted_audio_dir=input_original_audio_dir if input_original_audio_dir.is_dir() else None,
            project_output_diarized_json_dir=output_json_dir
        )
        print("\nStandalone LLMDiarizer test finished successfully.")
    except Exception as e:
        print(f"ERROR: LLMDiarizer failed during standalone execution: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 