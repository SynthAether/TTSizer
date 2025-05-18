import time
import os
import glob
import torch
import soundfile as sf
import librosa
import numpy as np
import yaml
from tqdm.auto import tqdm
from typing import Optional, Literal, List, Dict, Any
import torch.nn as nn
from ml_collections import ConfigDict
from pathlib import Path

from models.bs_roformer import MelBandRoformer
from utils import demix

import warnings
warnings.filterwarnings("ignore")

class VocalRemover:
    """Performs vocal separation on audio files using a pre-trained model."""
    def __init__(self, global_config: Dict[str, Any], vocal_remover_specific_config: Dict[str, Any]):
        """
        Initializes VocalRemover with settings from configuration.
        Args:
            global_config: Contains global project settings like model paths and project_setup.
            vocal_remover_specific_config: Contains settings specific to vocal removal.
        """
        # project_root is implicitly Path.cwd() due to orchestrator setting CWD
        
        self.model_path = Path(global_config['global_model_paths']['vocal_remover_model'])
        self.model_config_path = Path(global_config['global_model_paths']['vocal_remover_config'])
        
        self.model_type: str = vocal_remover_specific_config["model_type"]
        self.use_gpu: bool = vocal_remover_specific_config["use_gpu"]
        self.gpu_ids: Optional[List[int]] = vocal_remover_specific_config.get("gpu_ids", [0])
        self.output_format: str = vocal_remover_specific_config["output_format"].lower()
        self.output_pcm_type: str = vocal_remover_specific_config["output_pcm_type"]
        self.skip_if_output_exists: bool = vocal_remover_specific_config.get("skip_if_output_exists", True)
        
        self.current_project_output_vocals_dir: Optional[Path] = None 
        self.model_internal_config: Optional[ConfigDict] = None
        self.model: Optional[torch.nn.Module] = None
        self.device: Optional[torch.device] = None
        self.target_sr_from_model_config: int = 44100

        if self.output_format not in ['wav', 'flac']:
            print(f"Warning: Invalid output_format '{self.output_format}'. Defaulting to flac.")
            self.output_format = "flac"

        self._load_model_and_config()
        self._setup_device() 
        print("VocalRemover initialized.")

    def _load_model_and_config(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(f"VocalRemover model not found: {self.model_path.resolve()}")
        if not self.model_config_path.is_file():
            raise FileNotFoundError(f"VocalRemover model config not found: {self.model_config_path.resolve()}")

        print(f"VocalRemover: Loading model configuration from: {self.model_config_path.resolve()}")
        with open(self.model_config_path) as f:
            self.model_internal_config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
        
        self.target_sr_from_model_config = self.model_internal_config.audio.get('sample_rate', 44100)
            
        print(f"VocalRemover: Loading model architecture ({self.model_type})...")
        model_arch_params = dict(self.model_internal_config.model)
        if self.model_type == 'mel_band_roformer':
            self.model = MelBandRoformer(**model_arch_params)
        else:
            raise ValueError(f"Unsupported model_type for VocalRemover: {self.model_type}")

        print(f"VocalRemover: Loading model weights from: {self.model_path.resolve()}...")
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def _setup_device(self):
        if self.use_gpu and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            valid_ids = [i for i in self.gpu_ids if i >= 0 and i < num_gpus]
            if not valid_ids:
                print(f"VocalRemover Warning: GPU IDs {self.gpu_ids} invalid. Using CPU.")
                self.device = torch.device('cpu')
            elif len(valid_ids) == 1:
                self.device = torch.device(f'cuda:{valid_ids[0]}')
                print(f"VocalRemover: Using single GPU: cuda:{valid_ids[0]}")
            else:
                self.device = torch.device(f'cuda:{valid_ids[0]}') 
                self.model = nn.DataParallel(self.model, device_ids=valid_ids) 
                print(f"VocalRemover: Using {len(valid_ids)} GPUs with DataParallel: {valid_ids}")
        else:
            self.device = torch.device('cpu')
            if self.use_gpu:
                print("VocalRemover Warning: CUDA not available. Using CPU.")
            else:
                print("VocalRemover: Using CPU.")
        self.model.to(self.device)

    def _prepare_audio_for_model(self, mix: np.ndarray) -> np.ndarray:
        if mix.ndim == 1: 
            mix = np.expand_dims(mix, axis=0) 
        
        required_model_channels = self.model_internal_config.audio.get('num_channels', 1)
        
        if mix.shape[0] == 1 and required_model_channels == 2:
            mix = np.concatenate([mix, mix], axis=0) 
        elif mix.shape[0] > required_model_channels:
            mix = mix[:required_model_channels, :]
        return mix

    def _process_single_file(self, audio_file_path: Path):
        tqdm.write(f"  Processing: {audio_file_path.name}")

        try:
            mix_orig, sr_orig = librosa.load(audio_file_path, sr=None, mono=False)
            if sr_orig != self.target_sr_from_model_config:
                if mix_orig.ndim == 1: mix_orig = np.expand_dims(mix_orig, axis=0)
                mix = librosa.resample(mix_orig, orig_sr=sr_orig, target_sr=self.target_sr_from_model_config, res_type='kaiser_best')
            else:
                mix = mix_orig 
        except Exception as e:
            tqdm.write(f"    ERROR loading/resampling {audio_file_path.name}: {e}")
            return 
        
        mix = self._prepare_audio_for_model(mix)

        try:
            mix_tensor = torch.tensor(mix, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                 waveforms_dict = demix(self.model_internal_config, self.model, mix_tensor, self.device,
                                        model_type=self.model_type, pbar=False)
        except Exception as e:
            tqdm.write(f"    ERROR demixing {audio_file_path.name}: {e}")
            return 
        
        target_instrument = self.model_internal_config.training.get('target_instrument', 'vocals')
        vocal_key = target_instrument if target_instrument in waveforms_dict else None
        if not vocal_key and 'vocals' in waveforms_dict: vocal_key = 'vocals'

        if not vocal_key or vocal_key not in waveforms_dict:
            tqdm.write(f"    WARNING: Could not find '{target_instrument}' or 'vocals' in stems for {audio_file_path.name}. Stems: {list(waveforms_dict.keys())}")
            return

        vocals_data = waveforms_dict[vocal_key]
        output_filename = f"{audio_file_path.stem}_vocals.{self.output_format}"
        output_path = self.current_project_output_vocals_dir / output_filename

        subtype = self.output_pcm_type
        if self.output_format == 'wav' and subtype not in ['PCM_16', 'PCM_24', 'FLOAT']:
             tqdm.write(f"    Warning: Invalid pcm_type '{subtype}' for WAV. Defaulting to 'FLOAT'.")
             subtype = 'FLOAT'
        elif self.output_format == 'flac' and subtype not in ['PCM_16', 'PCM_24']:
             tqdm.write(f"    Warning: Invalid pcm_type '{subtype}' for FLAC. Defaulting to 'PCM_16'.")
             subtype = 'PCM_16' 

        try:
            data_to_write = vocals_data.T if vocals_data.ndim > 1 else vocals_data[:, np.newaxis]
            sf.write(str(output_path), data_to_write, self.target_sr_from_model_config, subtype=subtype)
        except Exception as e:
            tqdm.write(f"    ERROR writing {output_filename}: {e}")

    def run_separation_for_project(self, project_input_audio_dir: Path, project_output_vocals_base_dir: Path):
        self.current_project_output_vocals_dir = project_output_vocals_base_dir / "vocals"
        self.current_project_output_vocals_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nVocalRemover: Starting vocal separation for project.")
        print(f"  Input audio from: {project_input_audio_dir}")
        print(f"  Outputting vocals to: {self.current_project_output_vocals_dir}")
        if self.skip_if_output_exists:
            print("  Will skip files if output already exists.")

        audio_files_to_process = list(project_input_audio_dir.glob('*.flac')) 
        audio_files_to_process.extend(list(project_input_audio_dir.glob('*.wav')))
        for sub_dir in project_input_audio_dir.iterdir():
            if sub_dir.is_dir():
                audio_files_to_process.extend(list(sub_dir.glob('*.flac')))
                audio_files_to_process.extend(list(sub_dir.glob('*.wav')))
        
        audio_files_to_process = sorted(list(set(audio_files_to_process)))

        if not audio_files_to_process:
            print("VocalRemover: No compatible audio files (.flac, .wav) found to process.")
            return
            
        print(f"VocalRemover: Found {len(audio_files_to_process)} audio files to process.")

        skipped_count = 0
        processed_count = 0

        for audio_file_path in tqdm(audio_files_to_process, desc="Separating vocals", unit="file"):
            expected_output_filename = f"{audio_file_path.stem}_vocals.{self.output_format}"
            expected_output_path = self.current_project_output_vocals_dir / expected_output_filename

            if self.skip_if_output_exists and expected_output_path.exists():
                skipped_count += 1
                continue
            
            self._process_single_file(audio_file_path)
            processed_count +=1

        print(f"\nVocalRemover: Processing finished for project.")
        print(f"  Files processed: {processed_count}")
        print(f"  Files skipped (output existed): {skipped_count}")

# For standalone testing:
if __name__ == '__main__':
    config_file = Path("config.yaml")
    if not config_file.exists():
        config_file = Path("../config.yaml")
    
    if not config_file.exists():
        print("ERROR: config.yaml not found for standalone test.")
        exit(1)

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

    project_name = project_setup.get("project_name")
    output_base_abs = Path(project_setup.get("output_base_dir"))
    if not output_base_abs.is_absolute():
        output_base_abs = (project_root_for_test / output_base_abs).resolve()
    
    processing_project_dir_abs = output_base_abs / project_name

    # Input is the output of audio extraction (extract_audio_config.output_stage_folder_name / 'orig')
    extract_audio_cfg = cfg.get("extract_audio_config", {})
    extract_audio_output_folder = extract_audio_cfg.get("output_stage_folder_name", "01_extracted_audio_test")
    input_audio_dir = processing_project_dir_abs / extract_audio_output_folder / "orig"

    # Output base for this stage (vocal_removal_config.output_stage_folder_name)
    vocal_removal_cfg = cfg.get("vocal_removal_config")
    if not vocal_removal_cfg:
        print("ERROR: 'vocal_removal_config' not found in config.yaml.")
        exit(1)
    vocal_remover_output_folder = vocal_removal_cfg.get("output_stage_folder_name", "02_vocals_removed_STANDALONE_TEST")
    output_vocals_base_dir = processing_project_dir_abs / vocal_remover_output_folder

    if not input_audio_dir.is_dir():
        print(f"ERROR: Test input audio directory not found: {input_audio_dir}")
        print(f"Ensure stage '{extract_audio_output_folder}/orig' (or similar) from a previous test run exists.")
        exit(1)

    print(f"--- Running VocalRemover Standalone Test for project: {project_name} ---")
    print(f"Input audio from: {input_audio_dir}")
    print(f"Output vocals base to: {output_vocals_base_dir} (a 'vocals' subfolder will be created here)")

    try:
        # Pass the full config as global_config, and the specific stage config
        remover = VocalRemover(global_config=cfg, vocal_remover_specific_config=vocal_removal_cfg)
        remover.run_separation_for_project(
            project_input_audio_dir=input_audio_dir,
            project_output_vocals_base_dir=output_vocals_base_dir
        )
    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e}")
        print("Please check model paths in config.yaml (global_model_paths) and ensure they are correct relative to the project root.")
    except Exception as e_main:
        print(f"ERROR: An unexpected error occurred during standalone test: {e_main}")
        import traceback
        traceback.print_exc()
    
    print(f"--- Standalone Test Finished. Check output in: {output_vocals_base_dir / 'vocals'} ---")