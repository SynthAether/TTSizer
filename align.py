import warnings
warnings.filterwarnings("ignore")

import json
import os
import soundfile as sf
import re
import math
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModelForCTC, AutoTokenizer
import tempfile
import glob

from ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,
                                    postprocess_results, preprocess_text, load_audio)


# --------- Settings --------------
INPUT_JSON_DIR = "/home/taresh/Downloads/anime/audios/ChainsawMan/temp/gemini"
ORIGINAL_AUDIO_DIR = "/home/taresh/Downloads/anime/audios/ChainsawMan/vocals_normalized"
OUTPUT_BASE_DIR = "/home/taresh/Downloads/anime/audios/ChainsawMan/temp/final_clips"

TARGET_SPEAKERS = ['Makima', 'Power']

# -------------------------------------

SKIP_EPISODE_PATTERNS = []

# Processing Parameters
INIT_START_PAD_SEC = 0.4
INIT_END_PAD_SEC = 0.7
MIN_WORDS = 2
MIN_DURATION_SEC = 0.5 # Min original duration for non-expressions
ALIGNER_MODEL_PATH = "MahmoudAshraf/mms-300m-1130-forced-aligner"
ALIGNER_LANG = "en"
ALIGNER_BATCH_SIZE = 8
USE_GPU = True


def time_str_to_sec(time_str: str) -> float:
    parts = time_str.split(':')
    if len(parts) == 3: 
        m, s = map(float, parts[1:])
        if s >= 60.0: m += math.floor(s / 60.0); s %= 60.0; 
        return m * 60 + s
    if len(parts) == 2: 
        m, s = map(float, parts)
        if s >= 60.0: m += math.floor(s / 60.0); s %= 60.0; 
        return m * 60 + s
    if len(parts) == 1: return float(parts[0])
    raise ValueError(f"Invalid time format: {time_str}")

def sanitize_fn(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    return name.replace(' ', '_')

def load_aligner(device: str, model_path: str, dtype: torch.dtype) -> Tuple[AutoModelForCTC, AutoTokenizer]:
    print(f"Loading alignment model: {model_path}...")
    model = AutoModelForCTC.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Alignment model loaded.")
    return model, tokenizer

def get_speech_timestamps_in_chunk(
    audio_chunk_path: str, transcript: str, model, tokenizer,
    device: str, lang: str, batch_size: int
) -> Optional[Tuple[float, float]]:
    """Returns (start_sec, end_sec) of speech within the audio_chunk_path, or None."""
    if not transcript: return None
    try:
        wf = load_audio(audio_chunk_path, dtype=model.dtype, device=device)
        if wf is None or wf.nelement() == 0: return None

        with torch.no_grad():
            emissions, stride = generate_emissions(model, wf, batch_size=batch_size)
        tokens_s, text_s = preprocess_text(transcript, romanize=True, language=lang, split_size='word', star_frequency='edges')
        segments, scores, blank_tok = get_alignments(emissions, tokens_s, tokenizer)
        spans = get_spans(tokens_s, segments, blank_tok)
        word_ts = postprocess_results(text_s, spans, stride, scores)

        first_word, last_word = None, None
        for seg in word_ts:
            if seg.get('text') and seg['text'] != '<star>':
                if first_word is None: first_word = seg
                last_word = seg
        
        if first_word and last_word and 'start' in first_word and 'end' in last_word:
            start_rel, end_rel = first_word['start'], last_word['end']
            return (start_rel, end_rel) if end_rel > start_rel else None
        return None
    except Exception as e:
        print(f"  Alignment error for {os.path.basename(audio_chunk_path)}: {e}") 
        return None


def save_audio_segment(
    audio_data: np.ndarray, transcript: str, output_path: Path,
    samplerate: int, subtype: str
):
    """Saves the audio chunk and its transcript."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_data, samplerate, subtype=subtype)
    with open(output_path.with_suffix(".txt"), 'w', encoding='utf-8') as f:
        f.write(transcript)

def process_segment(
    seg_json: Dict,
    full_orig_audio: np.ndarray,
    audio_info: sf.SoundFile, # Contains samplerate, original_subtype, duration
    output_vocals_dir: Path,
    output_expr_dir: Path,
    temp_episode_dir: Path,
    spkr_counters: Dict[str, int],
    align_tools: Optional[Tuple[AutoModelForCTC, AutoTokenizer, str, str, int]], # (model, tok, device, lang, batch_size)
    params: Dict # Contains padding, min_words, min_duration
) -> Tuple[bool, bool, bool]: # (is_regular_saved, is_expr_saved, had_align_error)
    """Processes a single segment from JSON. Returns success flags."""
    spkr = seg_json.get('speaker')
    start_str = seg_json.get('start')
    end_str = seg_json.get('end')
    transcript = seg_json.get('transcript')

    # Basic validation
    if not (spkr and start_str and end_str and transcript): return False, False, False
    if spkr not in params['target_speakers']: return False, False, False

    try:
        t0_json = time_str_to_sec(start_str)
        t1_json = time_str_to_sec(end_str)
    except ValueError: return False, False, False # Invalid time format

    if t1_json <= t0_json: return False, False, False
    
    orig_dur = t1_json - t0_json
    is_expr = transcript.startswith(('(', '[')) and transcript.endswith((')', ']'))
    
    final_start_abs_sec = -1.0
    final_end_abs_sec = -1.0
    had_alignment_error = False

    if is_expr:
        final_start_abs_sec = t0_json
        final_end_abs_sec = t1_json
    else:
        transcript = transcript.strip()
        if len(transcript.split()) < params['min_words']: return False, False, False
        if orig_dur < params['min_duration_sec']: return False, False, False

        # Initial padded crop for alignment
        t0_padded = t0_json - params['init_start_pad']
        t1_padded = t1_json + params['init_end_pad']

        if t1_padded <= t0_padded: return False, False, False
        
        sr = audio_info.samplerate
        i0_padded = max(0, math.floor(t0_padded * sr))
        i1_padded = min(full_orig_audio.shape[0], math.ceil(t1_padded * sr))

        if i1_padded <= i0_padded: return False, False, False
        padded_chunk = full_orig_audio[i0_padded:i1_padded]
        if padded_chunk.size == 0: return False, False, False

        # Write temp file for alignment
        temp_wav_fn = f"temp_{sanitize_fn(spkr)}_{spkr_counters.get(spkr, 0)}.wav"
        temp_wav_path = temp_episode_dir / temp_wav_fn
        sf.write(str(temp_wav_path), padded_chunk, sr)

        # Align
        model, tok, device, lang, batch_size = align_tools
        aligned_times = get_speech_timestamps_in_chunk(
            str(temp_wav_path), transcript, model, tok, device, lang, batch_size
        )
        if temp_wav_path.exists(): temp_wav_path.unlink(missing_ok=True) # Clean up temp wav

        if aligned_times is None:
            had_alignment_error = True
            return False, False, had_alignment_error # Skip if alignment fails

        rel_start, rel_end = aligned_times
        final_start_abs_sec = t0_padded + rel_start
        final_end_abs_sec = t0_padded + rel_end

    # --- Final Cropping and Saving ---
    if final_start_abs_sec < 0 or final_end_abs_sec < 0 or final_end_abs_sec <= final_start_abs_sec:
        return False, False, had_alignment_error # Invalid final times

    # Ensure final times are within reasonable bounds of original audio
    final_start_abs_sec = max(0, final_start_abs_sec)
    final_end_abs_sec = min(audio_info.duration, final_end_abs_sec)
    if final_end_abs_sec <= final_start_abs_sec: return False, False, had_alignment_error

    sr = audio_info.samplerate
    i0_final = max(0, math.floor(final_start_abs_sec * sr))
    i1_final = min(full_orig_audio.shape[0], math.ceil(final_end_abs_sec * sr))

    if i1_final <= i0_final: return False, False, had_alignment_error
    final_chunk = full_orig_audio[i0_final:i1_final]
    if final_chunk.size == 0: return False, False, had_alignment_error

    # Save
    safe_spkr = sanitize_fn(spkr)
    spkr_counters[spkr] = spkr_counters.get(spkr, 0) + 1
    count = spkr_counters[spkr]
    base_fn = f"{safe_spkr}_{count:04d}.wav" # Output as WAV

    output_dir = output_expr_dir if is_expr else output_vocals_dir
    speaker_final_output_dir = output_dir / safe_spkr
    
    save_audio_segment(final_chunk, transcript, speaker_final_output_dir / base_fn, sr, audio_info.subtype)
    
    return not is_expr, is_expr, had_alignment_error


def process_episode_files(
    json_fpath: str, audio_fpath: str, output_base: Path,
    align_tools: Tuple, params: Dict
):
    """Loads data for a single episode and processes its segments."""
    ep_name = Path(json_fpath).stem
    tqdm.write(f"\n--- Processing Episode: {ep_name} ---")

    try:
        with open(json_fpath, 'r', encoding='utf-8') as f: segments_data = json.load(f)
        if not isinstance(segments_data, list): raise ValueError("JSON not list")
        
        audio_info = sf.info(str(audio_fpath)) # Get info before loading full audio
        full_audio, _ = sf.read(str(audio_fpath), dtype='float32', always_2d=False)
        if full_audio.ndim > 1 and full_audio.shape[1] > 1 : # Ensure mono
            print(f"Num channels: {full_audio.ndim} found for audio file: {str(audio_fpath)}, converting to mono")
            full_audio = np.mean(full_audio, axis=1)

    except Exception as e:
        tqdm.write(f"  Error loading data for {ep_name}: {e}. Skipping episode.")
        return 0, 0, 0, 0 # regular, expressions, json_skipped, align_skipped

    # Output dirs for this episode
    ep_vocals_out_dir = output_base / "vocals" / ep_name
    ep_expr_out_dir = output_base / "expressions" / ep_name

    # Temporary directory for this episode
    with tempfile.TemporaryDirectory(prefix=f"ep_temp_{ep_name}_") as temp_ep_dir_name:
        temp_episode_dir = Path(temp_ep_dir_name)

        spkr_counts_ep = defaultdict(int)
        saved_reg_ep, saved_expr_ep, skipped_json_ep, skipped_align_ep = 0, 0, 0, 0

        for seg_json in segments_data:
            reg_ok, expr_ok, align_err = process_segment(
                seg_json, full_audio, audio_info,
                ep_vocals_out_dir, ep_expr_out_dir, temp_episode_dir,
                spkr_counts_ep, align_tools, params
            )
            if reg_ok: saved_reg_ep += 1
            if expr_ok: saved_expr_ep += 1
            if align_err: skipped_align_ep +=1
            if not (reg_ok or expr_ok or align_err): # If no success and no align error, it was a json/filter skip
                skipped_json_ep +=1
    
    tqdm.write(f"  Finished {ep_name}. Regular: {saved_reg_ep}, Expressions: {saved_expr_ep}. "
               f"Skipped (Filter/JSON): {skipped_json_ep}, Skipped (Alignment): {skipped_align_ep}")
    return saved_reg_ep, saved_expr_ep, skipped_json_ep, skipped_align_ep


def run_batch_processing(
    json_input_dir: str, audio_input_dir: str, main_output_dir: str,
    target_speakers: List[str], skip_patterns: List[str],
    init_start_pad: float, init_end_pad: float,
    min_words: int, min_duration: float,
    aligner_model: str, aligner_lang: str, aligner_batch: int, use_gpu_flag: bool
):
    print("\n" + "="*40)
    print("Starting Batch Audio Segmentation and Alignment")
    print(f"Input JSONs: {json_input_dir}")
    print(f"Input Audios: {audio_input_dir}")
    print(f"Output Base: {main_output_dir}")
    print("="*40 + "\n")

    # Setup Device & Load Alignment Model
    device = "cuda" if use_gpu_flag and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    try:
        model_dtype = torch.float16 if device == "cuda" else torch.float32
        align_model, align_tok = load_aligner(device, aligner_model, model_dtype)
        align_tools = (align_model, align_tok, device, aligner_lang, aligner_batch)
    except Exception as e:
        print(f"FATAL: Could not load alignment model. Exiting. Error: {e}"); return

    json_files = sorted(glob.glob(os.path.join(json_input_dir, '*.json')))
    if not json_files: print(f"Error: No .json files found in '{json_input_dir}'."); return
    print(f"Found {len(json_files)} JSON files to process.")

    # Overall counters
    total_reg, total_expr, total_skip_json, total_skip_align = 0, 0, 0, 0
    eps_skipped_pattern, eps_skipped_no_audio = 0, 0

    # Collect parameters for process_segment
    processing_params = {
        'target_speakers': target_speakers,
        'init_start_pad': init_start_pad,
        'init_end_pad': init_end_pad,
        'min_words': min_words,
        'min_duration_sec': min_duration
    }

    for json_fpath_str in tqdm(json_files, desc="Episodes"):
        ep_name = Path(json_fpath_str).stem
        if any(pattern in ep_name for pattern in skip_patterns):
            tqdm.write(f"\n--- Skipping Episode (pattern): {ep_name} ---")
            eps_skipped_pattern += 1; continue

        audio_fpath = Path(audio_input_dir) / f"{ep_name}.flac" # Assuming FLAC
        if not audio_fpath.is_file():
            tqdm.write(f"  Audio not found for {ep_name} at {audio_fpath}. Skipping.")
            eps_skipped_no_audio += 1; continue
        
        reg, expr, skip_j, skip_a = process_episode_files(
            json_fpath_str, str(audio_fpath), Path(main_output_dir),
            align_tools, processing_params
        )
        total_reg += reg; total_expr += expr; total_skip_json += skip_j; total_skip_align += skip_a

    print("\n" + "="*40)
    print("Batch Processing Complete.")
    print(f"Total Episodes (JSONs): {len(json_files)}")
    print(f"Skipped by Pattern: {eps_skipped_pattern}, Skipped (No Audio): {eps_skipped_no_audio}")
    print(f"Total Regular Clips: {total_reg}, Total Expression Clips: {total_expr}")
    print(f"Total Segments Skipped (Filter/JSON): {total_skip_json}")
    print(f"Total Segments Skipped (Alignment Fail): {total_skip_align}")
    print("="*40)


if __name__ == '__main__':
    if not os.path.isdir(INPUT_JSON_DIR): print(f"ERROR: JSON dir not found: {INPUT_JSON_DIR}")
    elif not os.path.isdir(ORIGINAL_AUDIO_DIR): print(f"ERROR: Audio dir not found: {ORIGINAL_AUDIO_DIR}")
    elif not TARGET_SPEAKERS: print(f"ERROR: TARGET_SPEAKERS list empty.")
    else:
        run_batch_processing(
            json_input_dir=INPUT_JSON_DIR,
            audio_input_dir=ORIGINAL_AUDIO_DIR,
            main_output_dir=OUTPUT_BASE_DIR,
            target_speakers=TARGET_SPEAKERS,
            skip_patterns=SKIP_EPISODE_PATTERNS,
            init_start_pad=INIT_START_PAD_SEC,
            init_end_pad=INIT_END_PAD_SEC,
            min_words=MIN_WORDS,
            min_duration=MIN_DURATION_SEC,
            aligner_model=ALIGNER_MODEL_PATH,
            aligner_lang=ALIGNER_LANG,
            aligner_batch=ALIGNER_BATCH_SIZE,
            use_gpu_flag=USE_GPU
        )