import time
import os
import glob
import torch
import soundfile as sf
import librosa
import numpy as np
import yaml
from tqdm.auto import tqdm
from typing import Optional, Literal, List
import torch.nn as nn
from ml_collections import ConfigDict

from models.bs_roformer import MelBandRoformer
from utils import demix 

import warnings
warnings.filterwarnings("ignore")

SAMPLING_RATE = 44100

class VocalSeparator:
    """
    A class to perform vocal separation on audio files (WAV or FLAC) within
    a folder using a pre-trained MelBandRoformer model.
    Supports single or multi-GPU processing. Outputs separated vocals losslessly.
    """
    def __init__(self,
                 model_path: str,
                 config_path: str,
                 output_dir: str,
                 model_type: str = 'mel_band_roformer',
                 use_gpu: bool = True,
                 gpu_ids: Optional[List[int]] = [0],
                 output_format: Literal['wav', 'flac'] = 'flac',
                 output_pcm_type: Literal['PCM_16', 'PCM_24', 'FLOAT'] = 'PCM_24'):

        print("Initializing Vocal Separator...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config path not found: {config_path}")

        self.model_path = model_path
        self.config_path = config_path
        self.output_dir = output_dir
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.output_format = output_format.lower()
        self.output_pcm_type = output_pcm_type

        if self.output_format not in ['wav', 'flac']:
            raise ValueError("output_format must be 'wav' or 'flac'")

        self._load_config_and_model()
        self._setup_device()

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory set to: {self.output_dir}")
        print(f"Output format: {self.output_format.upper()} ({self.output_pcm_type})")
        print("Initialization complete.")

    def _load_config_and_model(self):
        """Loads the configuration file and the model."""
        print(f"Loading configuration from: {self.config_path}")
        load_start_time = time.time()
        try:
            with open(self.config_path) as f:
                self.config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            
            print(f"Loading model architecture ({self.model_type})...")
            model_config_dict = dict(self.config.model) 
            self.model = MelBandRoformer(**model_config_dict)

            print(f"Loading model weights from: {self.model_path}...")
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device('cpu'))
            )
            self.model.eval()
            print(f"Model loading took: {time.time() - load_start_time:.2f} seconds.")

        except Exception as e:
            print(f"An unexpected error occurred during model/config loading: {e}")
            raise

    def _setup_device(self):
        """Sets up the computation device (CPU or specific GPU(s))."""
        if self.use_gpu and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            # Validate provided GPU IDs
            valid_ids = [i for i in self.gpu_ids if i >= 0 and i < num_gpus]
            if not valid_ids:
                print(f"Warning: Provided GPU IDs {self.gpu_ids} are invalid or unavailable (found {num_gpus} GPUs). Falling back to CPU.")
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
            elif len(valid_ids) == 1:
                self.device = torch.device(f'cuda:{valid_ids[0]}')
                self.model = self.model.to(self.device)
                print(f"Using specified single GPU: cuda:{valid_ids[0]}")
            else:
                self.device = torch.device(f'cuda:{valid_ids[0]}') # Primary device
                # Important: Wrap model *before* moving it if DataParallel handles placement
                self.model = nn.DataParallel(self.model, device_ids=valid_ids) 
                self.model.to(self.device) # Move the wrapped model
                print(f"Using {len(valid_ids)} GPUs with DataParallel: {valid_ids}")
        else:
            # Use CPU
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
            if self.use_gpu:
                print("CUDA not available, using CPU.")
            else:
                print("Using CPU as requested.")


    def _prepare_audio(self, mix: np.ndarray) -> np.ndarray:
        """Checks audio channels and converts mono to stereo if needed by config."""
        if mix.ndim == 1:
            mix = np.expand_dims(mix, axis=0) 
        
        required_channels = self.config.audio.get('num_channels', 1) 
        
        if mix.shape[0] == 1 and required_channels == 2:
            print("Input audio is mono, but model configuration expects stereo. Duplicating channel.")
            mix = np.concatenate([mix, mix], axis=0) 
        elif mix.shape[0] > 2:
             print(f"Warning: Input audio has {mix.shape[0]} channels. Taking only the first two.")
             mix = mix[:2, :] 

        return mix


    def _process_file(self, file_path: str):
        """Loads, processes, and saves the vocals for a single audio file."""
        print(f"\nProcessing: {os.path.basename(file_path)}")
        
        target_sr = self.config.audio.get('sample_rate', SAMPLING_RATE)

        try:
            mix_orig, sr_orig = librosa.load(file_path, sr=None, mono=False)

            # Resample if needed AFTER loading
            if sr_orig != target_sr:
                print(f"  Resampling from {sr_orig} Hz to {target_sr} Hz...")
                # Ensure mix_orig is 2D before resampling multichannel
                if mix_orig.ndim == 1:
                    mix_orig = np.expand_dims(mix_orig, axis=0)
                mix = librosa.resample(mix_orig, orig_sr=sr_orig, target_sr=target_sr, res_type='kaiser_best')
                print(f"  Done resampling")
            else:
                mix = mix_orig 

        except Exception as e:
            print(f'--> ERROR: Cannot read track: {os.path.basename(file_path)}')
            print(f'     Error message: {str(e)}')
            return 
        
        mix = self._prepare_audio(mix)

        try:
            mix_tensor = torch.tensor(mix, dtype=torch.float32).to(self.device)
            
            # Use torch.no_grad() for inference efficiency
            with torch.no_grad():
                 waveforms_dict = demix(self.config, self.model, mix_tensor, self.device, 
                                        model_type=self.model_type, pbar=True) # Disable inner progress bar if tqdm used outside

        except Exception as e:
            print(f"--> ERROR: Failed to demix track: {os.path.basename(file_path)}")
            print(f"     Error message: {str(e)}")
            return 
        
        instruments = self.config.training.get('instruments', [])
        target_instrument = self.config.training.get('target_instrument', None)
        
        vocal_key = None
        if target_instrument is not None and target_instrument in waveforms_dict:
            vocal_key = target_instrument
        elif 'vocals' in waveforms_dict:
            vocal_key = 'vocals'
        elif instruments and 'vocals' in instruments and 'vocals' in waveforms_dict:
             vocal_key = 'vocals' 

        if not vocal_key:
            print(f"--> WARNING: Could not find 'vocals' or specified target instrument "
                  f"('{target_instrument}') in separated stems for {os.path.basename(file_path)}. Skipping save.")
            print(f"    Available stems: {list(waveforms_dict.keys())}")
            return

        estimates_numpy = waveforms_dict[vocal_key]

        file_name_base = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{file_name_base}_vocals.{self.output_format}"
        output_path = os.path.join(self.output_dir, output_filename)

        subtype = self.output_pcm_type
        if self.output_format == 'wav' and subtype not in ['PCM_16', 'PCM_24', 'FLOAT']:
             print(f"Warning: Invalid pcm_type '{subtype}' for WAV output. Defaulting to 'FLOAT'.")
             subtype = 'FLOAT'
        elif self.output_format == 'flac' and subtype not in ['PCM_16', 'PCM_24']:
             print(f"Warning: Invalid pcm_type '{subtype}' for FLAC output. Defaulting to 'PCM_16'.")
             subtype = 'PCM_16' 

        try:
            if estimates_numpy.ndim == 1: 
                 data_to_write = estimates_numpy[:, np.newaxis] 
            else: 
                 data_to_write = estimates_numpy.T 
            
            sf.write(output_path, data_to_write, target_sr, subtype=subtype)
            print(f"--> Saved vocals to: {output_filename}")
        except Exception as e:
            print(f"--> ERROR: Failed to write output file: {output_filename}")
            print(f"     Error message: {str(e)}")


    def process_folder(self, input_folder: str, verbose: bool = False):
        """
        Processes all compatible audio files in the specified input folder.

        Args:
            input_folder: Path to the folder containing audio files (.wav).
            verbose: If True, prints more detailed logs.
        """
        if not os.path.isdir(input_folder):
             print(f"Error: Input folder not found or is not a directory: {input_folder}")
             return

        start_time = time.time()
        print(f"\nStarting processing for folder: {input_folder}")
        
        all_mixtures_path = glob.glob(os.path.join(input_folder, '*.flac')) 
        total_tracks = len(all_mixtures_path)

        if total_tracks == 0:
            print("No .flac files found in the input folder.")
            return
            
        print(f"Found {total_tracks} '.flac' tracks to process.")

        # Use tqdm for overall progress
        for file_path in tqdm(all_mixtures_path, desc="Processing audio files"):
            self._process_file(file_path)

        print(f"\nFolder processing finished. Elapsed time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    # --- Configuration ---
    MODEL_PATH = "weights/kimmel_unwa_ft2_bleedless.ckpt" 
    CONFIG_PATH = "configs/config_kimmel_unwa_ft.yaml"
    INPUT_FOLDER = "/home/taresh/Downloads/anime/audios/Frieren/orig"
    OUTPUT_FOLDER = "/home/taresh/Downloads/anime/audios/Frieren/vocals"
    
    USE_GPU = True 
    GPU_IDS = [0]
                   
    OUTPUT_FORMAT = 'flac' 
    OUTPUT_PCM_TYPE = 'PCM_24'
    MODEL_TYPE = 'mel_band_roformer' 

    separator = VocalSeparator(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        output_dir=OUTPUT_FOLDER,
        model_type=MODEL_TYPE,
        use_gpu=USE_GPU,
        gpu_ids=GPU_IDS,
        output_format=OUTPUT_FORMAT,
        output_pcm_type=OUTPUT_PCM_TYPE
    )

    separator.process_folder(INPUT_FOLDER)

    print("\nProcessing completed successfully!")