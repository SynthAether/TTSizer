import warnings
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr
import soundfile as sf
import torch
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import shutil
import yaml
from typing import Dict, Any, Tuple
from ttsizer.utils.logger import get_logger

logger = get_logger("asr_processor")

class ASRProcessor:
    """Processes audio files using a Parakeet ASR model for transcription and flagging.

    This class transcribes audio files, extracts word-level timestamps, and flags segments
    where the detected speech boundaries deviate significantly from the file boundaries.
    Flagged segments are saved with padding for further review.
    """

    def __init__(self, global_config: Dict[str, Any], asr_config: Dict[str, Any]):
        """Initializes the ASRProcessor with global and ASR-specific configurations.

        Args:
            global_config: Dictionary containing global project setup information.
            asr_config: Dictionary containing configuration specific to the ASR processor.
        """
        self.target_speakers = global_config['project_setup']['target_speaker_labels']
        
        self.model_name = asr_config["model_name"]
        self.batch_size = asr_config["batch_size"]
        self.device = asr_config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.time_thresh = asr_config["timestamp_deviation_threshold_sec"]
        self.padding = asr_config["padding_sec"]

        self.flagged_dir = asr_config["flagged_output_folder"]
        self.def_dir = "definite"

        self.model = None
        self._load_model()
        logger.info("ParakeetASRProcessor initialized.")

    def _load_model(self):
        """Loads the pre-trained ASR model from Nemo."""
        logger.info(f"Loading ASR model: {self.model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name, 
            map_location=self.device
        )
        self.model.eval()

    def _get_times(self, result: Any, dur: float) -> Tuple[float, float]:
        """Extracts start and end times from ASR result word timestamps, clamped to duration.

        Args:
            result: The ASR transcription result object, potentially containing timestamps.
            dur: The total duration of the audio file in seconds.

        Returns:
            A tuple (start_time, end_time) in seconds for the transcribed speech.
        """
        start, end = 0.0, dur
        if hasattr(result, 'timestamp') and isinstance(result.timestamp, dict) and 'word' in result.timestamp:
            words = result.timestamp['word']
            valid = [s for s in words if isinstance(s, dict) and 'start' in s and 'end' in s]
            if valid:
                print("valid")
                start = max(0.0, min(s['start'] for s in valid))
                end = min(dur, max(s['end'] for s in valid))
                if end < start:
                    start, end = 0.0, dur
        return start, end

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Processes all .wav files in speaker-specific subdirectories for ASR.

        Transcribes files, checks for timestamp deviations, and saves flagged files
        (with their transcripts) to a separate 'flagged' output directory.

        Args:
            input_dir: The base input directory containing speaker subdirectories
                       (e.g., .../outlier_detector_output/SPEAKER_NAME/definite/).
            output_dir: The base output directory where ASR results, including a
                        'flagged' subdirectory for each speaker, will be saved.
        """
        for spkr in self.target_speakers:
            spkr_dir = spkr.replace(" ", "_")
            
            in_dir = input_dir / spkr_dir / self.def_dir
            flagged_dir = output_dir / spkr_dir / self.flagged_dir
            if flagged_dir.exists(): 
                shutil.rmtree(flagged_dir)
            flagged_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Input: {in_dir.resolve()}")
            logger.info(f"Flagged output: {flagged_dir.resolve()}")

            files = sorted(list(in_dir.rglob("*.wav")))
            if not files:
                logger.warning(f"No '.wav' files found in {in_dir}")
                return

            logger.info(f"Found {len(files)} files")
            flagged = 0

            for i in tqdm(range(0, len(files), self.batch_size), desc="Processing", unit="batch"):
                batch = [str(p) for p in files[i:i + self.batch_size]]
                batch_files = files[i:i + self.batch_size]

                try:
                    results = self.model.transcribe(
                        batch,
                        batch_size=len(batch),
                        timestamps=True,
                        verbose=False
                    )
                except Exception as e:
                    logger.error(f"Error in batch: {e}")
                    continue

                if isinstance(results, tuple) and len(results) == 1 and isinstance(results[0], list):
                    results = results[0]
                elif not isinstance(results, list):
                    continue

                if len(results) != len(batch_files):
                    continue

                for path, result in zip(batch_files, results):
                    try:
                        audio, sr = sf.read(str(path), dtype='float32')
                        if audio.ndim > 1:
                            audio = np.mean(audio, axis=1)
                        dur = len(audio) / sr if sr > 0 else 0.0
                        
                        text = ""
                        start, end = 0.0, dur

                        if isinstance(result, str):
                            text = result
                        elif hasattr(result, 'text'):
                            text = result.text or ""
                            start, end = self._get_times(result, dur)

                        # Check if needs flagging
                        start_dev = abs(start - 0.0)
                        end_dev = abs(end - dur)
                        
                        if start_dev > self.time_thresh or end_dev > self.time_thresh:
                            pad_start = max(0.0, start - self.padding)
                            pad_end = min(dur, end + self.padding)
                            
                            if pad_end <= pad_start:
                                if end > start:
                                    pad_start, pad_end = start, end
                                else:
                                    pad_start, pad_end = 0.0, dur

                            start_frame = int(pad_start * sr)
                            end_frame = int(pad_end * sr)

                            crop_audio = audio
                            if dur > 0 and end_frame > start_frame and start_frame >= 0 and end_frame <= len(audio):
                                crop_audio = audio[start_frame:end_frame]
                            
                            if crop_audio.size == 0:
                                if audio.size > 0:
                                    crop_audio = audio
                                else:
                                    continue
                            
                            flagged_audio = flagged_dir / path.name
                            flagged_txt = flagged_dir / path.with_suffix(".txt").name
                            sf.write(str(flagged_audio), crop_audio, sr, subtype='PCM_24')
                            with open(flagged_txt, 'w', encoding='utf-8') as f:
                                f.write(text)
                            flagged += 1
                    
                    except Exception as e:
                        logger.error(f"Error processing {path.name}: {e}")

            logger.info(f"\nProcessing complete for {spkr}")
            logger.info(f"Total files: {len(files)}")
            logger.info(f"Flagged: {flagged}")

if __name__ == "__main__":
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    base_dir = Path(cfg["project_setup"]["output_base_dir"]) / Path(cfg["project_setup"]["series_name"])
    in_dir = base_dir / Path(cfg["outlier_detector"]["output_folder"])
    out_dir = base_dir / Path(cfg["asr_processor"]["output_folder"])
    
    print(f"Processing ASR for {in_dir}")
    processor = ASRProcessor(cfg, cfg["asr_processor"])
    processor.process_directory(in_dir, out_dir)