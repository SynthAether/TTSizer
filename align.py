import warnings
warnings.filterwarnings("ignore")

import json
import os
import soundfile as sf
import re
import math
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModelForCTC, AutoTokenizer
import tempfile
import glob
import yaml # For loading config in __main__
from dotenv import load_dotenv # For API keys and other env vars

# Assuming ctc_forced_aligner is in the Python path or installed
from ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,
                                    postprocess_results, preprocess_text, load_audio)

class AudioTranscriptAligner:
    """Aligns transcripts to audio segments using a CTC Forced Aligner model."""
    def __init__(self, global_config: Dict[str, Any], aligner_config: Dict[str, Any]):
        # self.project_root = Path(global_config["project_settings"]["project_root_dir"]) # Removed, CWD is project root
        
        # Aligner model settings
        self.aligner_model_path: str = aligner_config["aligner_model_name_or_path"]
        self.aligner_lang: str = aligner_config["language_code"]
        self.aligner_batch_size: int = aligner_config["batch_size"]
        self.use_gpu: bool = aligner_config["use_gpu"]
        
        # Processing parameters
        self.target_speakers: List[str] = aligner_config.get("target_speakers_of_interest", []) # Can be empty
        self.init_start_pad_sec: float = aligner_config["init_start_pad_seconds"]
        self.init_end_pad_sec: float = aligner_config["init_end_pad_seconds"]
        self.min_words: int = aligner_config["min_words_per_segment"]
        self.min_duration_sec: float = aligner_config["min_duration_seconds_segment"]
        self.skip_episode_patterns: List[str] = aligner_config.get("skip_episode_patterns", [])
        
        # Output settings
        self.output_audio_format: str = aligner_config.get("output_audio_format", "wav").lower()
        # Ensure subtype is valid for sf.write, common ones are PCM_16, PCM_24, FLOAT
        self.output_audio_subtype: str = aligner_config.get("output_audio_subtype", "PCM_24") 
        self.skip_if_output_exists: bool = aligner_config.get("skip_if_output_exists", True) # Episode-level skip for now

        self.device: torch.device = self._setup_device()
        self.model: Optional[AutoModelForCTC] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._load_alignment_model()
        
        tqdm.write("AudioTranscriptAligner initialized.")

    def _setup_device(self) -> torch.device:
        if self.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda:0") # Simple single GPU selection
            tqdm.write(f"AudioTranscriptAligner: Using device: {device}")
        else:
            device = torch.device("cpu")
            tqdm.write(f"AudioTranscriptAligner: Using CPU (GPU not requested or not available).")
        return device

    def _load_alignment_model(self):
        tqdm.write(f"AudioTranscriptAligner: Loading alignment model: {self.aligner_model_path}...")
        try:
            model_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
            self.model = AutoModelForCTC.from_pretrained(self.aligner_model_path, torch_dtype=model_dtype).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.aligner_model_path)
            tqdm.write("AudioTranscriptAligner: Alignment model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load alignment model \'{self.aligner_model_path}\'. Error: {e}")

    def _time_str_to_sec(self, time_str: str) -> float:
        parts = time_str.split(':')
        try:
            if len(parts) == 3: 
                h_or_m, m_or_s, s_val = float(parts[0]), float(parts[1]), float(parts[2])
                if h_or_m >= 60 or (len(parts[0]) > 2 and parts[0] != "00") : 
                    return h_or_m * 3600 + m_or_s * 60 + s_val
                else: 
                    m, s = h_or_m, m_or_s 
                    actual_seconds = s + s_val # s is whole seconds, s_val is fractional part if it was mm:ss.mmm, OR actual seconds if it was mm:ss:ss
                    if actual_seconds >= 60: m += math.floor(actual_seconds/60); actual_seconds %=60
                    return m * 60 + actual_seconds
            elif len(parts) == 2: # MM:SS.mmm
                m, s = float(parts[0]), float(parts[1])
                if s >= 60.0: m += math.floor(s / 60.0); s %= 60.0
                return m * 60 + s
            elif len(parts) == 1: # SS.mmm (Reinstated)
                return float(parts[0])
        except ValueError:
            pass 
        raise ValueError(f"Invalid time format for CTC Aligner: '{time_str}'. Expected HH:MM:SS.mmm, MM:SS.mmm or SS.mmm.")

    def _sanitize_filename_part(self, name: str) -> str:
        name = re.sub(r'[\\\\/*?:\"<>|]', '', name) 
        return name.replace(' ', '_')

    def _get_speech_timestamps_in_chunk(self, audio_chunk_path: Path, transcript: str) -> Optional[Tuple[float, float]]:
        if not transcript or not self.model or not self.tokenizer:
            return None
        try:
            wf = load_audio(str(audio_chunk_path), dtype=self.model.dtype, device=self.device)
            if wf is None or wf.nelement() == 0:
                return None

            with torch.no_grad():
                emissions, stride = generate_emissions(self.model, wf, batch_size=self.aligner_batch_size)
            
            tokens_s, text_s = preprocess_text(transcript, romanize=True, language=self.aligner_lang, split_size='word', star_frequency='edges')
            segments, scores, blank_tok = get_alignments(emissions, tokens_s, self.tokenizer)
            spans = get_spans(tokens_s, segments, blank_tok)
            word_ts = postprocess_results(text_s, spans, stride, scores)

            first_word, last_word = None, None
            for seg_dict in word_ts:
                if isinstance(seg_dict, dict) and seg_dict.get('text') and seg_dict['text'] != '<star>':
                    if first_word is None: first_word = seg_dict
                    last_word = seg_dict
        
            if first_word and last_word and 'start' in first_word and 'end' in last_word:
                start_rel, end_rel = first_word['start'], last_word['end']
                if end_rel > start_rel:
                    return (start_rel, end_rel)
            return None
        except Exception as e:
            tqdm.write(f"    Alignment processing error for {audio_chunk_path.name}: {type(e).__name__} - {e}") 
            return None

    def _save_aligned_audio_segment(self, audio_data: np.ndarray, transcript: str, output_path: Path, samplerate: int):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_data, samplerate, subtype=self.output_audio_subtype)
        with open(output_path.with_suffix(".txt"), 'w', encoding='utf-8') as f:
            f.write(transcript)

    def _process_single_segment(
        self, seg_json: Dict[str, Any], full_orig_audio_data: np.ndarray, 
        audio_info: Any, 
        output_base_for_episode: Path, temp_episode_dir: Path,
        spkr_counters: Dict[str, int], episode_name_for_log: str
    ) -> Tuple[bool, bool, bool]: 
        
        spkr = seg_json.get('speaker')
        start_str = seg_json.get('start')
        end_str = seg_json.get('end')
        transcript = seg_json.get('transcript')

        if not (spkr and start_str and end_str):
            return False, False, False 
        
        if self.target_speakers and spkr not in self.target_speakers and spkr != "SOUND":
            return False, False, False 

        try:
            t0_json = self._time_str_to_sec(start_str)
            t1_json = self._time_str_to_sec(end_str)
        except ValueError as e:
            tqdm.write(f"    Skipping segment in {episode_name_for_log} for '{spkr}' due to invalid time: {e}")
            return False, False, False

        if t1_json <= t0_json: return False, False, False
        
        orig_dur = t1_json - t0_json
        is_expr_or_sound = (spkr == "SOUND") or \
                           (isinstance(transcript, str) and transcript.startswith(('(', '[')) and transcript.endswith((')', ']')))
        
        final_start_abs_sec, final_end_abs_sec = -1.0, -1.0
        alignment_error_occurred = False
        current_transcript_for_saving = transcript if isinstance(transcript, str) else ""

        if is_expr_or_sound:
            final_start_abs_sec, final_end_abs_sec = t0_json, t1_json
        else:
            current_transcript_for_saving = current_transcript_for_saving.strip()
            if not current_transcript_for_saving: return False, False, False 
            if len(current_transcript_for_saving.split()) < self.min_words: return False, False, False
            if orig_dur < self.min_duration_sec: return False, False, False

            t0_padded = max(0.0, t0_json - self.init_start_pad_sec)
            t1_padded = min(audio_info.duration, t1_json + self.init_end_pad_sec)

        if t1_padded <= t0_padded: return False, False, False
        
        sr = audio_info.samplerate
        i0_padded, i1_padded = math.floor(t0_padded * sr), math.ceil(t1_padded * sr)

        if i1_padded <= i0_padded: return False, False, False
        padded_chunk = full_orig_audio_data[i0_padded:i1_padded]
        if padded_chunk.size == 0: return False, False, False

        temp_wav_fn = f"temp_align_{self._sanitize_filename_part(spkr)}_{spkr_counters.get(spkr,0)}_{t0_json:.3f}.wav"
        temp_wav_path = temp_episode_dir / temp_wav_fn
        sf.write(str(temp_wav_path), padded_chunk, sr, subtype=self.output_audio_subtype)

        aligned_times = self._get_speech_timestamps_in_chunk(temp_wav_path, current_transcript_for_saving)
        if temp_wav_path.exists(): temp_wav_path.unlink(missing_ok=True)

        if aligned_times is None:
            alignment_error_occurred = True
            return False, False, alignment_error_occurred

        rel_start, rel_end = aligned_times
        final_start_abs_sec = t0_padded + rel_start
        final_end_abs_sec = t0_padded + rel_end

        if not (0 <= final_start_abs_sec < final_end_abs_sec <= audio_info.duration):
            return False, False, alignment_error_occurred

        sr = audio_info.samplerate
        i0_final, i1_final = math.floor(final_start_abs_sec * sr), math.ceil(final_end_abs_sec * sr)

        if i1_final <= i0_final: return False, False, alignment_error_occurred
        final_chunk = full_orig_audio_data[i0_final:i1_final]
        if final_chunk.size == 0: return False, False, alignment_error_occurred

        speaker_label_for_path = self._sanitize_filename_part(spkr)
        spkr_counters[speaker_label_for_path] = spkr_counters.get(speaker_label_for_path, 0) + 1
        count = spkr_counters[speaker_label_for_path]
        base_fn = f"{speaker_label_for_path}_{self._sanitize_filename_part(episode_name_for_log)}_{count:05d}.{self.output_audio_format}"
        
        if is_expr_or_sound:
            segment_output_dir = output_base_for_episode / "expressions_sound"
        else:
            segment_output_dir = output_base_for_episode / "vocals" / speaker_label_for_path
        
        output_file_path = segment_output_dir / base_fn

        if self.skip_if_output_exists and output_file_path.exists() and output_file_path.with_suffix(".txt").exists():
            return not is_expr_or_sound, is_expr_or_sound, alignment_error_occurred 

        self._save_aligned_audio_segment(final_chunk, current_transcript_for_saving, output_file_path, sr)
        return not is_expr_or_sound, is_expr_or_sound, alignment_error_occurred

    def _run_alignment_for_single_episode(
        self, episode_llm_json_path: Path, episode_original_audio_path: Path,
        episode_name: str, 
        project_output_aligned_clips_base_dir: Path 
    ) -> Tuple[int, int, int, int]: 
        
        tqdm.write(f"\n--- Processing Episode: {episode_name} ---")
        tqdm.write(f"  JSON: {episode_llm_json_path.name}")
        tqdm.write(f"  Audio: {episode_original_audio_path.name}")

        try:
            with open(episode_llm_json_path, 'r', encoding='utf-8') as f: segments_data = json.load(f)
            if not isinstance(segments_data, list): 
                tqdm.write(f"    ERROR: JSON in {episode_llm_json_path.name} is not a list. Skipping.")
                return 0, 0, 0, 0
            # Make sure segments_data is a list of dicts as expected by downstream logic
            if not all(isinstance(item, dict) for item in segments_data):
                tqdm.write(f"    ERROR: Segments data in {episode_llm_json_path.name} is not a list of dictionaries. Skipping.")
                return 0, 0, 0, 0

            if not episode_original_audio_path.exists():
                tqdm.write(f"    ERROR: Original audio file not found: {episode_original_audio_path}. Skipping episode.")
                return 0, 0, 0, 0
            
            try:
                audio_data, sr = sf.read(str(episode_original_audio_path), dtype='float32')
                audio_info = sf.info(str(episode_original_audio_path))
            except Exception as e:
                tqdm.write(f"    ERROR: Could not read audio file {episode_original_audio_path.name}: {e}. Skipping episode.")
                return 0, 0, 0, 0

            output_base_for_episode = project_output_aligned_clips_base_dir / episode_name
            # output_base_for_episode.mkdir(parents=True, exist_ok=True) # Handled by _save_aligned_audio_segment
            
            # Temporary directory for this episode's intermediate files (e.g. for CTC aligner chunks)
            temp_episode_dir = Path(tempfile.mkdtemp(prefix=f"align_temp_{episode_name}_"))

            num_good_segments, num_expr_sound_segments, num_skipped, num_alignment_errors = 0, 0, 0, 0
            spkr_counters: Dict[str, int] = defaultdict(int)

            for seg_idx, seg_json in enumerate(tqdm(segments_data, desc=f"  Segments in {episode_name}", leave=False)):
                if not isinstance(seg_json, dict):
                    tqdm.write(f"    Skipping invalid segment (not a dict) at index {seg_idx} in {episode_name}")
                    num_skipped += 1
                    continue

                is_good, is_expr_sound, align_err = self._process_single_segment(
                    seg_json, audio_data, audio_info, 
                    output_base_for_episode, temp_episode_dir,
                    spkr_counters, episode_name
                )
                if is_good: num_good_segments += 1
                if is_expr_sound: num_expr_sound_segments +=1
                if align_err: num_alignment_errors +=1
                if not is_good and not is_expr_sound and not align_err: num_skipped +=1

            # Clean up temporary directory
            try:
                # for item in temp_episode_dir.iterdir(): item.unlink()
                # temp_episode_dir.rmdir()
                # Using shutil.rmtree for robustness with non-empty dirs
                import shutil
                shutil.rmtree(temp_episode_dir)
            except Exception as e_clean:
                tqdm.write(f"    Warning: Could not remove temp directory {temp_episode_dir}: {e_clean}")

            tqdm.write(f"  Finished {episode_name}: {num_good_segments} vocal segments, {num_expr_sound_segments} expr/sound, {num_skipped} skipped, {num_alignment_errors} align errors.")
            return num_good_segments, num_expr_sound_segments, num_skipped, num_alignment_errors
        except json.JSONDecodeError as e:
            tqdm.write(f"    ERROR: Failed to decode JSON from {episode_llm_json_path.name}: {e}. Skipping episode.")
            return 0,0,0,0
        except FileNotFoundError as e:
            tqdm.write(f"    ERROR: File not found during processing of {episode_name}: {e}. Skipping episode.")
            return 0,0,0,0
        except Exception as e_outer:
            tqdm.write(f"    UNEXPECTED ERROR during processing of {episode_name}: {type(e_outer).__name__} - {e_outer}. Skipping episode.")
            import traceback
            traceback.print_exc()
            return 0,0,0,0

    def run_alignment_for_project(
        self,
        project_llm_json_input_dir: Path,    
        project_normalized_audio_input_dir: Path, 
        project_name: str,                     
        project_output_parent_dir: Path,     
        stage_output_folder_name: str       
    ):
        tqdm.write("\n" + "="*50)
        tqdm.write(f"AudioTranscriptAligner: Starting Batch Alignment for Project: '{project_name}'")
        tqdm.write(f"  Input LLM JSONs from: {project_llm_json_input_dir.resolve()}")
        tqdm.write(f"  Input Normalized Audio from: {project_normalized_audio_input_dir.resolve()}")
        
        final_project_output_destination = project_output_parent_dir / project_name / stage_output_folder_name
        tqdm.write(f"  Output Aligned Clips to: {final_project_output_destination.resolve()}")
        tqdm.write("="*50 + "\n")

        json_files = sorted(list(project_llm_json_input_dir.rglob('*.json')))
        if not json_files:
            tqdm.write(f"ERROR: No .json files found in '{project_llm_json_input_dir}'. Cannot proceed."); return

        tqdm.write(f"Found {len(json_files)} JSON files (potential episodes) for '{project_name}'.")
        stats = defaultdict(int)
        processed_ep_count = 0

        for llm_json_file_path in tqdm(json_files, desc=f"Aligning Episodes for {project_name}", unit="ep"):
            episode_stem = llm_json_file_path.stem
            if any(pattern in episode_stem for pattern in self.skip_episode_patterns):
                tqdm.write(f"-- Skipping episode {episode_stem} due to skip_episode_patterns.")
                stats['episodes_skipped_pattern'] += 1; continue

            audio_file_path = None
            for ext in [".flac", ".wav"]:
                potential_audio_path = project_normalized_audio_input_dir / (episode_stem + ext)
                if potential_audio_path.is_file():
                    audio_file_path = potential_audio_path; break
            
            if not audio_file_path:
                tqdm.write(f"  WARNING: No corresponding audio found for '{llm_json_file_path.name}\' in '{project_normalized_audio_input_dir}\'. Skipping.")
                stats['episodes_skipped_no_audio'] += 1; continue
            
            ep_voc, ep_expr, ep_json_skip, ep_align_skip = self._run_alignment_for_single_episode(
                llm_json_file_path, audio_file_path,
                episode_stem, final_project_output_destination 
            )
            stats['total_vocals_clips'] += ep_voc
            stats['total_expression_clips'] += ep_expr
            stats['total_segments_skipped_json_filter'] += ep_json_skip
            stats['total_segments_skipped_alignment_errors'] += ep_align_skip
            if ep_voc > 0 or ep_expr > 0: processed_ep_count += 1

        tqdm.write("\n" + "="*50)
        tqdm.write(f"AudioTranscriptAligner: Batch Alignment Complete for Project '{project_name}'.")
        tqdm.write(f"  Total JSONs found: {len(json_files)}")
        tqdm.write(f"  Episodes effectively processed (produced clips): {processed_ep_count}")
        tqdm.write(f"  Episodes skipped (pattern): {stats['episodes_skipped_pattern']}")
        tqdm.write(f"  Episodes skipped (no audio): {stats['episodes_skipped_no_audio']}")
        tqdm.write(f"  Total Regular Vocal Clips Saved: {stats['total_vocals_clips']}")
        tqdm.write(f"  Total Expression/Sound Clips Saved: {stats['total_expression_clips']}")
        tqdm.write(f"  Total Segments Skipped (JSON/Filter): {stats['total_segments_skipped_json_filter']}")
        tqdm.write(f"  Total Segments Skipped (Alignment Err): {stats['total_segments_skipped_alignment_errors']}")
        tqdm.write("="*50)

# For standalone testing:
if __name__ == '__main__':
    config_file = Path("config.yaml")
    if not config_file.exists():
        # Try parent directory if running from a 'scripts' subdirectory
        config_file = Path("../config.yaml") 
    
    if not config_file.exists():
        print(f"ERROR: {config_file.name} not found in current or parent directory for standalone test.")
        exit(1)

    # Load environment variables from .env file if present
    dotenv_path = config_file.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        # print(f"Loaded environment variables from: {dotenv_path}")
    # else:
        # print(f"Info: No .env file found at {dotenv_path}. Relying on globally set environment variables.")

    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Could not load or parse {config_file.resolve()}: {e}")
        exit(1)

    # Set CWD to project root (where config.yaml is)
    project_root_for_test = config_file.parent.resolve()
    if Path.cwd() != project_root_for_test:
        # print(f"Standalone test: Changing CWD to project root: {project_root_for_test}")
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

    # Aligner config
    aligner_cfg = cfg.get("align_audio_transcript_config")
    if not aligner_cfg:
        print("ERROR: 'align_audio_transcript_config' not found in config.yaml.")
        exit(1)

    # Input: LLM Diarized JSON (from llm_diarizer_config.output_stage_folder_name)
    llm_diarizer_config = cfg.get("llm_diarizer_config", {})
    llm_json_input_folder = llm_diarizer_config.get("output_stage_folder_name", "04_llm_diarized_json_test")
    input_llm_json_dir = processing_project_dir_abs / llm_json_input_folder

    # Input: Normalized Audio (from normalize_audio_config.output_stage_folder_name)
    norm_audio_cfg = cfg.get("normalize_audio_config", {})
    norm_audio_output_folder = norm_audio_cfg.get("output_stage_folder_name", "03_vocals_normalized_test")
    input_normalized_audio_dir = processing_project_dir_abs / norm_audio_output_folder

    # Output: Aligned Clips (align_audio_transcript_config.output_stage_folder_name)
    aligner_output_folder = aligner_cfg.get("output_stage_folder_name", "05_aligned_clips_STANDALONE_TEST")
    output_aligned_clips_dir = processing_project_dir_abs / aligner_output_folder

    if not input_llm_json_dir.is_dir():
        print(f"ERROR: Test input LLM JSON directory not found: {input_llm_json_dir}")
        # exit(1) # Allow to proceed to catch in class if it handles it, or if only one input is truly critical
    if not input_normalized_audio_dir.is_dir():
        print(f"ERROR: Test input normalized audio directory not found: {input_normalized_audio_dir}")
        # exit(1)
    
    # Ensure output directory exists for the test
    output_aligned_clips_dir.mkdir(parents=True, exist_ok=True) 

    print(f"\n--- Running AudioTranscriptAligner Standalone Test for project: {project_name_for_test} ---")
    print(f"Input LLM JSON from: {input_llm_json_dir}")
    print(f"Input Normalized Audio from: {input_normalized_audio_dir}")
    print(f"Output Aligned Clips to: {output_aligned_clips_dir}")
    print(f"Using GPU if available: {aligner_cfg.get('use_gpu', False)}")

    # Initialize AudioTranscriptAligner
    try:
        aligner = AudioTranscriptAligner(global_config=cfg, aligner_config=aligner_cfg)
    except Exception as e:
        print(f"ERROR: Failed to initialize AudioTranscriptAligner: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Run the alignment process
    try:
        aligner.run_alignment_for_project(
            project_llm_json_input_dir=input_llm_json_dir,
            project_normalized_audio_input_dir=input_normalized_audio_dir,
            project_name=project_name_for_test,
            project_output_parent_dir=processing_project_dir_abs, 
            stage_output_folder_name=aligner_output_folder 
        )
        print("\nStandalone AudioTranscriptAligner test finished successfully.")
    except Exception as e:
        print(f"ERROR: AudioTranscriptAligner failed during standalone execution: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"--- Standalone Test Finished. Check output in: {output_aligned_clips_dir} ---")