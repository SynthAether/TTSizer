#!/usr/bin/env python3
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

# Ensure the script directory is in the Python path for module imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ttsizer.core.extract_audio import AudioExtractor
from ttsizer.core.vocal_removal import VocalRemover
from ttsizer.core.normalize import AudioNormalizer
from ttsizer.core.llm_call import LLMDiarizer
from ttsizer.core.align import AudioTranscriptAligner
from ttsizer.core.outlier import OutlierDetector
from ttsizer.core.parakeet import ParakeetASRProcessor
from ttsizer.utils.logger import get_logger

# Initialize logger for the orchestrator
logger = get_logger("orchestrator")

# Define the order of pipeline stages
STAGES = [
    "extract_audio",
    "vocal_removal",
    "normalize_audio",
    "llm_diarizer",
    "ctc_aligner",
    "find_outliers",
    "parakeet_asr"
]

class PipelineOrchestrator:
    def __init__(self, cfg_path: str = "config.yaml"):
        self.cfg_path = Path(cfg_path).resolve()
        self.root = self.cfg_path.parent

        with open(self.cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.setup = self.cfg["project_setup"]
        self.name = self.setup["project_name"]
        self.vid_dir = Path(self.setup.get("video_source_parent_dir", "./video_sources"))
        self.out_dir = Path(self.setup.get("output_base_dir", "./processing_output"))

        if not self.vid_dir.is_absolute():
            self.vid_dir = (self.root / self.vid_dir).resolve()
        if not self.out_dir.is_absolute():
            self.out_dir = (self.root / self.out_dir).resolve()
            
        self.proj_dir = (self.out_dir / self.name).resolve()

        if Path.cwd() != self.root:
            os.chdir(self.root)
        
        self.runners = {
            STAGES[0]: self._run_extract,
            STAGES[1]: self._run_vocal,
            STAGES[2]: self._run_norm,
            STAGES[3]: self._run_llm,
            STAGES[4]: self._run_align,
            STAGES[5]: self._run_outlier,
            STAGES[6]: self._run_asr,
        }

    def _get_cfg(self, stage: str) -> Dict[str, Any]:
        return self.cfg.get(f"{stage}_config", {})

    def _get_paths(self, stage: str, prev: Optional[str] = None, in_sub: Optional[str] = None, out_sub: Optional[str] = None) -> Tuple[Optional[Path], Path]:
        cfg = self._get_cfg(stage)
        out_folder = cfg.get("output_stage_folder_name")
        if not out_folder:
            out_folder = f"{stage}_output"
            logger.warning(f"No output folder for {stage}, using {out_folder}")

        out = self.proj_dir / out_folder
        if out_sub:
            out = out / out_sub
        
        inp = None
        if prev:
            prev_cfg = self._get_cfg(prev)
            prev_folder = prev_cfg.get("output_stage_folder_name")
            if prev_folder:
                inp = self.proj_dir / prev_folder
                if in_sub:
                    inp = inp / in_sub
        
        return inp, out

    def _run_extract(self):
        stage = STAGES[0]
        logger.info(f"\nRunning {stage}")
        cfg = self._get_cfg(stage)
        if not cfg: return

        vid_dir = self.vid_dir / self.name
        _, out = self._get_paths(stage)

        extractor = AudioExtractor(cfg)
        extractor.run(vid_dir, out)
        logger.info(f"{stage} complete")

    def _run_vocal(self):
        stage = STAGES[1]
        logger.info(f"\nRunning {stage}")
        cfg = self._get_cfg(stage)
        if not cfg: return
        
        inp, out = self._get_paths(stage, STAGES[0], "orig")
        if not inp: return

        remover = VocalRemover(self.cfg, cfg)
        remover.run(inp, out)
        logger.info(f"{stage} complete")

    def _run_norm(self):
        stage = STAGES[2]
        logger.info(f"\nRunning {stage}")
        cfg = self._get_cfg(stage)
        if not cfg: return
        
        inp, out = self._get_paths(stage, STAGES[1], "vocals")
        if not inp: return
        
        normalizer = AudioNormalizer(cfg)
        normalizer.run(inp, out)
        logger.info(f"{stage} complete")

    def _run_llm(self):
        stage = STAGES[3]
        logger.info(f"\nRunning {stage}")
        cfg = self._get_cfg(stage)
        if not cfg: return
        
        inp, out = self._get_paths(stage, STAGES[2])
        if not inp: return

        orig_cfg = self._get_cfg(STAGES[0])
        orig_folder = orig_cfg.get("output_stage_folder_name")
        orig_dir = self.proj_dir / orig_folder / "orig" if orig_folder else None

        diarizer = LLMDiarizer(self.cfg, cfg)
        diarizer.run(inp, orig_dir, out)
        logger.info(f"{stage} complete")

    def _run_align(self):
        stage = STAGES[4]
        logger.info(f"\nRunning {stage}")
        cfg = self._get_cfg(stage)
        if not cfg: return

        inp, _ = self._get_paths(stage, STAGES[3])
        if not inp: return

        norm_cfg = self._get_cfg(STAGES[2])
        norm_folder = norm_cfg.get("output_stage_folder_name")
        if not norm_folder: return
        norm_dir = self.proj_dir / norm_folder

        aligner = AudioTranscriptAligner(self.cfg, cfg)
        aligner.run(inp, norm_dir, self.name, self.proj_dir)
        logger.info(f"{stage} complete")

    def _run_outlier(self):
        stage = STAGES[5]
        logger.info(f"\nRunning {stage}")
        cfg = self._get_cfg(stage)
        if not cfg: return

        inp, out = self._get_paths(stage, STAGES[4])
        if not inp: return

        detector = OutlierDetector(self.cfg, cfg)
        detector.run(self.name, cfg["audio_sources"], inp, out)
        logger.info(f"{stage} complete")

    def _run_asr(self):
        stage = STAGES[6]
        logger.info(f"\nRunning {stage}")
        cfg = self._get_cfg(stage)
        if not cfg: return

        inp, out = self._get_paths(stage, STAGES[5])
        if not inp: return

        processor = ParakeetASRProcessor(self.cfg, cfg)
        processor.run(inp, out)
        logger.info(f"{stage} complete")

    def run(self):
        for stage in STAGES:
            if stage in self.runners:
                self.runners[stage]()

def main():
    # Allow overriding config path via environment variable, otherwise default to "ttsizer/configs/config.yaml"
    cfg_path = os.getenv("TTSIZER_CONFIG", "ttsizer/configs/config.yaml")
    print(f"Starting pipeline with config: {cfg_path}")
    
    try:
        orchestrator = PipelineOrchestrator(cfg_path)
        orchestrator.run()
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
