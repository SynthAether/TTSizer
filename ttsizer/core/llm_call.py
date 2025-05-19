import warnings
warnings.filterwarnings("ignore")

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
import yaml
import google.generativeai as genai
from google.generativeai import types as genai_types
import time
from ttsizer.utils.logger import get_logger

logger = get_logger("llm_diarizer")

class LLMDiarizer:
    """Handles diarization and transcription using a large language model (LLM).
    
    Uses Gemini model to analyze audio files and generate speaker diarization.
    Supports retry logic for API calls and handles JSON output formatting.
    """
    def __init__(self, global_config: Dict[str, Any], diarizer_config: Dict[str, Any]):
        self.proj = global_config["project_setup"]
        self.name = self.proj["project_name"]
        self.speakers = self.proj["target_speaker_labels_for_diarization"]

        self.model_name = diarizer_config["model_name"]
        self.temp = diarizer_config["temperature"]
        self.top_p = diarizer_config["top_p"]
        self.tmpl_file = Path(diarizer_config["prompt_template_file"])
        self.skip = diarizer_config.get("skip_if_output_exists", True)

        self.model = genai.GenerativeModel(self.model_name)
        self.tmpl = ""
        if self.tmpl_file.exists():
            with open(self.tmpl_file, 'r', encoding='utf-8') as f:
                self.tmpl = f.read()
        else:
            raise FileNotFoundError(f"Template not found: {self.tmpl_file}")

    def _gen_content(self, model_input: List[Any], max_retries: int = 3, delay: int = 5) -> Optional[str]:
        retries = 0
        while retries < max_retries:
            try:
                resp = self.model.generate_content(model_input)
                if resp.candidates and resp.candidates[0].content.parts:
                    return resp.candidates[0].content.parts[0].text
                return None
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    return None
                time.sleep(delay)
                delay *= 2
        return None

    def _process_file(self, norm_path: Path, orig_path: Optional[Path], out_path: Path):
        if self.skip and out_path.exists():
            return

        audio = genai.upload_file(path=str(norm_path))
        chars = ", ".join(self.speakers) if self.speakers else "Not Specified"
        char1 = self.speakers[0] if self.speakers else "First Main Character"
        
        prompt = self.tmpl.replace("{ANIME_TITLE}", self.name)
        prompt = prompt.replace("{CHARACTERS_OF_INTEREST}", chars)
        prompt = prompt.replace("{CHARACTER_1}", char1)

        text = self._gen_content([prompt, audio])
        genai.delete_file(audio.name)

        if not text:
            return

        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            debug_path = out_path.with_suffix(".raw_llm_output.txt")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(text)

    def run_diarization_for_project(self, norm_dir: Path, orig_dir: Optional[Path], out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(norm_dir.glob("*.flac")))
        if not files:
            files = sorted(list(norm_dir.glob("*.wav")))
        
        if not files:
            return

        for norm_path in tqdm(files, desc="Processing Episodes", unit="episode"):
            out_path = out_dir / norm_path.with_suffix(".json").name
            
            orig_path = None
            if orig_dir:
                flac = orig_dir / norm_path.with_suffix(".flac").name
                wav = orig_dir / norm_path.with_suffix(".wav").name
                if flac.exists():
                    orig_path = flac
                elif wav.exists():
                    orig_path = wav

            self._process_file(norm_path, orig_path, out_path)
        

if __name__ == '__main__':
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    os.chdir(Path("configs/config.yaml").parent)
    
    norm_dir = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["project_name"] / cfg["normalize_audio_config"]["output_stage_folder_name"]
    out_dir = Path(cfg["project_setup"]["output_base_dir"]) / cfg["project_setup"]["project_name"] / cfg["llm_diarizer_config"]["output_stage_folder_name"]
    
    print(f"Testing diarization for {norm_dir}")
    diarizer = LLMDiarizer(global_config=cfg, diarizer_config=cfg["llm_diarizer_config"])
    diarizer.run_diarization_for_project(norm_dir, None, out_dir) 