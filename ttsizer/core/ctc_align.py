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
import yaml
from ttsizer.utils.logger import get_logger
from ttsizer.core.ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,
                               postprocess_results, preprocess_text, load_audio)

logger = get_logger("aligner")

class AudioTranscriptAligner:
    """Aligns audio segments with their transcripts using CTC forced alignment.
    
    Uses a pre-trained model to precisely align speech segments with their transcripts,
    handling multiple speakers and supporting both vocal and sound effect segments.
    """
    def __init__(self, global_config: Dict[str, Any], aligner_config: Dict[str, Any]):
        self.proj = global_config["project_setup"]
        self.name = self.proj["project_name"]
        self.speakers = self.proj["target_speaker_labels_for_diarization"]

        self.model_path = aligner_config["aligner_model_name_or_path"]
        self.lang = aligner_config["language_code"]
        self.batch_size = aligner_config["batch_size"]
        self.use_gpu = aligner_config["use_gpu"]
        
        self.target_spkrs = aligner_config.get("target_speakers_of_interest", [])
        self.start_pad = aligner_config["init_start_pad_seconds"]
        self.end_pad = aligner_config["init_end_pad_seconds"]
        self.min_words = aligner_config["min_words_per_segment"]
        self.min_dur = aligner_config["min_duration_seconds_segment"]
        self.skip_patterns = aligner_config.get("skip_episode_patterns", [])
        
        self.out_fmt = aligner_config.get("output_audio_format", "wav").lower()
        self.out_subtype = aligner_config.get("output_audio_subtype", "PCM_24")
        self.skip_existing = aligner_config.get("skip_if_output_exists", True)

        self.device = torch.device("cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        model_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.model = AutoModelForCTC.from_pretrained(self.model_path, torch_dtype=model_dtype).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def _time_to_sec(self, time_str: str) -> float:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            if h >= 60 or (len(parts[0]) > 2 and parts[0] != "00"):
                return h * 3600 + m * 60 + s
            else:
                s_total = s + m
                if s_total >= 60:
                    m += math.floor(s_total/60)
                    s_total %= 60
                return m * 60 + s_total
        elif len(parts) == 2:
            m, s = float(parts[0]), float(parts[1])
            if s >= 60:
                m += math.floor(s/60)
                s %= 60
            return m * 60 + s
        elif len(parts) == 1:
            return float(parts[0])
        raise ValueError(f"Invalid time format: {time_str}")

    def _clean_name(self, name: str) -> str:
        return re.sub(r'[\\\\/*?:\"<>|]', '', name).replace(' ', '_')

    def _get_timestamps(self, audio_path: Path, text: str) -> Optional[Tuple[float, float]]:
        if not text or not self.model or not self.tokenizer:
            return None

        wf = load_audio(str(audio_path), dtype=self.model.dtype, device=self.device)
        if wf is None or wf.nelement() == 0:
            return None

        with torch.no_grad():
            emissions, stride = generate_emissions(self.model, wf, batch_size=self.batch_size)
        
        tokens, text_s = preprocess_text(text, romanize=True, language=self.lang, split_size='word', star_frequency='edges')
        segments, scores, blank = get_alignments(emissions, tokens, self.tokenizer)
        spans = get_spans(tokens, segments, blank)
        word_ts = postprocess_results(text_s, spans, stride, scores)

        first = next((seg for seg in word_ts if isinstance(seg, dict) and seg.get('text') and seg['text'] != '<star>'), None)
        last = next((seg for seg in reversed(word_ts) if isinstance(seg, dict) and seg.get('text') and seg['text'] != '<star>'), None)
        
        if first and last and 'start' in first and 'end' in last:
            start, end = first['start'], last['end']
            return (start, end) if end > start else None
        return None

    def _save_segment(self, audio: np.ndarray, text: str, out_path: Path, sr: int):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, sr, subtype=self.out_subtype)
        with open(out_path.with_suffix(".txt"), 'w', encoding='utf-8') as f:
            f.write(text)

    def _process_segment(
        self, seg: Dict[str, Any], audio: np.ndarray, info: Any, 
        out_dir: Path, temp_dir: Path, counters: Dict[str, int], ep_name: str
    ) -> Tuple[bool, bool, bool]:
        
        spkr = seg.get('speaker')
        start = seg.get('start')
        end = seg.get('end')
        text = seg.get('transcript')

        if not (spkr and start and end):
            return False, False, False 
        
        if self.target_spkrs and spkr not in self.target_spkrs and spkr != "SOUND":
            return False, False, False 

        t0 = self._time_to_sec(start)
        t1 = self._time_to_sec(end)
        if t1 <= t0:
            return False, False, False
        
        dur = t1 - t0
        is_sound = (spkr == "SOUND") or (isinstance(text, str) and text.startswith(('(', '[')) and text.endswith((')', ']')))
        
        start_sec, end_sec = -1.0, -1.0
        align_err = False
        text = text if isinstance(text, str) else ""

        if is_sound:
            start_sec, end_sec = t0, t1
        else:
            text = text.strip()
            if not text or len(text.split()) < self.min_words or dur < self.min_dur:
                return False, False, False

            t0_pad = max(0.0, t0 - self.start_pad)
            t1_pad = min(info.duration, t1 + self.end_pad)
            if t1_pad <= t0_pad:
                return False, False, False
            
            sr = info.samplerate
            i0, i1 = math.floor(t0_pad * sr), math.ceil(t1_pad * sr)
            if i1 <= i0:
                return False, False, False

            chunk = audio[i0:i1]
            if chunk.size == 0:
                return False, False, False

            temp_path = temp_dir / f"temp_align_{self._clean_name(spkr)}_{counters.get(spkr,0)}_{t0:.3f}.wav"
            sf.write(str(temp_path), chunk, sr, subtype=self.out_subtype)

            times = self._get_timestamps(temp_path, text)
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

            if times is None:
                align_err = True
                return False, False, align_err

            rel_start, rel_end = times
            start_sec = t0_pad + rel_start
            end_sec = t0_pad + rel_end

            if not (0 <= start_sec < end_sec <= info.duration):
                return False, False, align_err

        sr = info.samplerate
        i0, i1 = math.floor(start_sec * sr), math.ceil(end_sec * sr)
        if i1 <= i0:
            return False, False, align_err

        final_chunk = audio[i0:i1]
        if final_chunk.size == 0:
            return False, False, align_err

        spkr_name = self._clean_name(spkr)
        counters[spkr_name] = counters.get(spkr_name, 0) + 1
        count = counters[spkr_name]
        fn = f"{spkr_name}_{self._clean_name(ep_name)}_{count:05d}.{self.out_fmt}"
        
        out_subdir = out_dir / ("expressions_sound" if is_sound else f"vocals/{spkr_name}")
        out_path = out_subdir / fn

        if self.skip_existing and out_path.exists() and out_path.with_suffix(".txt").exists():
            return not is_sound, is_sound, align_err 

        self._save_segment(final_chunk, text, out_path, sr)
        return not is_sound, is_sound, align_err

    def _process_episode(
        self, json_path: Path, audio_path: Path, ep_name: str, out_dir: Path
    ) -> Tuple[int, int, int, int]:
        
        logger.info(f"\n--- Processing Episode: {ep_name} ---")
        logger.info(f"  JSON: {json_path.name}")
        logger.info(f"  Audio: {audio_path.name}")

        with open(json_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)

        audio, info = sf.read(str(audio_path))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            counters = defaultdict(int)
            vocal_count = sound_count = 0
            align_errors = 0

            for seg in tqdm(segments, desc="Processing Segments", unit="segment"):
                is_vocal, is_sound, has_error = self._process_segment(
                    seg, audio, info, out_dir, temp_dir, counters, ep_name
                )
                if is_vocal:
                    vocal_count += 1
                if is_sound:
                    sound_count += 1
                if has_error:
                    align_errors += 1

        return vocal_count, sound_count, align_errors, len(segments)

    def run_alignment_for_project(
        self, json_dir: Path, audio_dir: Path, proj_name: str, out_dir: Path, stage_name: str
    ):
        out_dir = out_dir / proj_name / stage_name
        out_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(list(json_dir.glob("*.json")))
        if not json_files:
            return

        total_vocals = total_sounds = total_errors = total_segments = 0
        for json_path in tqdm(json_files, desc="Processing Episodes", unit="episode"):
            ep_name = json_path.stem
            if any(pattern in ep_name for pattern in self.skip_patterns):
                continue

            audio_path = None
            for fmt in ['.flac', '.wav']:
                path = audio_dir / json_path.with_suffix(fmt).name
                if path.exists():
                    audio_path = path
                    break

            if not audio_path:
                continue

            vocals, sounds, errors, segments = self._process_episode(
                json_path, audio_path, ep_name, out_dir
            )
            total_vocals += vocals
            total_sounds += sounds
            total_errors += errors
            total_segments += segments

        logger.info(f"\n=== Alignment Summary ===")
        logger.info(f"Total Segments: {total_segments}")
        logger.info(f"Vocal Segments: {total_vocals}")
        logger.info(f"Sound Segments: {total_sounds}")
        logger.info(f"Alignment Errors: {total_errors}")


if __name__ == '__main__':
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    os.chdir(Path("configs/config.yaml").parent)
    
    json_dir = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["project_name"] / cfg["llm_diarizer_config"]["output_stage_folder_name"]
    audio_dir = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["project_name"] / cfg["normalize_audio_config"]["output_stage_folder_name"]
    out_dir = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["project_name"] / cfg["aligner_config"]["output_stage_folder_name"]
    
    print(f"Testing alignment for {json_dir}")
    aligner = AudioTranscriptAligner(global_config=cfg, aligner_config=cfg["aligner_config"])
    aligner.run_alignment_for_project(json_dir, audio_dir, cfg["project_setup"]["project_name"], out_dir, cfg["aligner_config"]["output_stage_folder_name"]) 