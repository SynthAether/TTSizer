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
from pathlib import Path

from models.bs_roformer import MelBandRoformer
from utils import demix

import warnings
warnings.filterwarnings("ignore")

SKIP_IF_OUTPUT_EXISTS = True 

# Model settings
SAMPLING_RATE = 44100
MODEL_PATH = "weights/kimmel_unwa_ft2_bleedless.ckpt"
CONFIG_PATH = "configs/config_kimmel_unwa_ft.yaml"
OUTPUT_FORMAT = 'flac'
OUTPUT_PCM_TYPE = 'PCM_24'
MODEL_TYPE = 'mel_band_roformer'


class VocalSeparator:
    """
    A class to perform vocal separation on audio files (WAV or FLAC) within
    a folder using a pre-trained MelBandRoformer model.
    Supports single or multi-GPU processing. Outputs separated vocals losslessly.
    Can skip processing if output file already exists.
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
        self.base_output_dir = Path(output_dir)
        self.output_dir = self.base_output_dir
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        self.output_format = output_format.lower()
        self.output_pcm_type = output_pcm_type

        if self.output_format not in ['wav', 'flac']:
            raise ValueError("output_format must be 'wav' or 'flac'")

        self._load_config_and_model()
        self._setup_device()

        print(f"Base output directory structure will be relative to: {self.base_output_dir}")
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
                self.device = torch.device(f'cuda:{valid_ids[0]}') 
                self.model = nn.DataParallel(self.model, device_ids=valid_ids) 
                self.model.to(self.device) 
                print(f"Using {len(valid_ids)} GPUs with DataParallel: {valid_ids}")
        else:
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
            tqdm.write("  Input audio is mono, but model configuration expects stereo. Duplicating channel.")
            mix = np.concatenate([mix, mix], axis=0) 
        elif mix.shape[0] > 2:
             tqdm.write(f"  Warning: Input audio has {mix.shape[0]} channels. Taking only the first two.")
             mix = mix[:2, :] 
        return mix


    def _process_file(self, file_path: str):
        """Loads, processes, and saves the vocals for a single audio file.
        Assumes self.output_dir is set correctly for the current batch.
        Prints messages using tqdm.write.
        """
        audio_file_path_obj = Path(file_path)
        tqdm.write(f"\nProcessing: {audio_file_path_obj.name}") # \n for visual separation in log
        
        target_sr = self.config.audio.get('sample_rate', SAMPLING_RATE)

        try:
            mix_orig, sr_orig = librosa.load(file_path, sr=None, mono=False)

            if sr_orig != target_sr:
                tqdm.write(f"  Resampling from {sr_orig} Hz to {target_sr} Hz...")
                if mix_orig.ndim == 1:
                    mix_orig = np.expand_dims(mix_orig, axis=0)
                mix = librosa.resample(mix_orig, orig_sr=sr_orig, target_sr=target_sr, res_type='kaiser_best')
                tqdm.write(f"  Done resampling")
            else:
                mix = mix_orig 
        except Exception as e:
            tqdm.write(f'--> ERROR: Cannot read track: {audio_file_path_obj.name}')
            tqdm.write(f'     Error message: {str(e)}')
            return 
        
        mix = self._prepare_audio(mix)

        try:
            mix_tensor = torch.tensor(mix, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                 waveforms_dict = demix(self.config, self.model, mix_tensor, self.device, 
                                        model_type=self.model_type, pbar=False) # Disable inner progress bar
        except Exception as e:
            tqdm.write(f"--> ERROR: Failed to demix track: {audio_file_path_obj.name}")
            tqdm.write(f"     Error message: {str(e)}")
            return 
        
        instruments = self.config.training.get('instruments', [])
        target_instrument = self.config.training.get('target_instrument', None)
        
        vocal_key = None
        if target_instrument is not None and target_instrument in waveforms_dict:
            vocal_key = target_instrument
        elif 'vocals' in waveforms_dict:
            vocal_key = 'vocals'
        elif instruments and 'vocals' in instruments and 'vocals' in waveforms_dict: # Check if 'vocals' is a listed instrument
             vocal_key = 'vocals' 

        if not vocal_key:
            tqdm.write(f"--> WARNING: Could not find 'vocals' or specified target instrument "
                  f"('{target_instrument}') in separated stems for {audio_file_path_obj.name}. Skipping save.")
            tqdm.write(f"    Available stems: {list(waveforms_dict.keys())}")
            return

        estimates_numpy = waveforms_dict[vocal_key]

        file_name_base = audio_file_path_obj.stem
        output_filename = f"{file_name_base}_vocals.{self.output_format}"
        output_path = self.output_dir / output_filename


        subtype = self.output_pcm_type
        if self.output_format == 'wav' and subtype not in ['PCM_16', 'PCM_24', 'FLOAT']:
             tqdm.write(f"Warning: Invalid pcm_type '{subtype}' for WAV output. Defaulting to 'FLOAT'.")
             subtype = 'FLOAT'
        elif self.output_format == 'flac' and subtype not in ['PCM_16', 'PCM_24']:
             tqdm.write(f"Warning: Invalid pcm_type '{subtype}' for FLAC output. Defaulting to 'PCM_16'.")
             subtype = 'PCM_16' 

        try:
            if estimates_numpy.ndim == 1: 
                 data_to_write = estimates_numpy[:, np.newaxis] 
            else: 
                 data_to_write = estimates_numpy.T 
            
            sf.write(str(output_path), data_to_write, target_sr, subtype=subtype)
            tqdm.write(f"--> Saved vocals to: {output_filename} in {self.output_dir.name}")
        except Exception as e:
            tqdm.write(f"--> ERROR: Failed to write output file: {output_filename}")
            tqdm.write(f"     Error message: {str(e)}")


    def process_folder(self, input_folder: str, verbose: bool = False): # verbose not actively used yet
        """
        Processes all compatible audio files in the specified input folder.
        Skips files if SKIP_IF_OUTPUT_EXISTS is True and output already exists.
        self.output_dir must be set to the correct output path for this folder *before* calling.
        """
        input_folder_path = Path(input_folder)
        if not input_folder_path.is_dir():
             tqdm.write(f"Error: Input folder not found or is not a directory: {input_folder_path}")
             return

        start_time_folder = time.time()
        tqdm.write(f"\n--- Starting processing for folder: {input_folder_path.name} ---")
        tqdm.write(f"Input: {input_folder_path}")
        tqdm.write(f"Output: {self.output_dir}")
        
        if SKIP_IF_OUTPUT_EXISTS:
            tqdm.write("SKIP_IF_OUTPUT_EXISTS is True. Will skip files with existing outputs.")
        else:
            tqdm.write("SKIP_IF_OUTPUT_EXISTS is False. Will process all files.")
        
        all_mixtures_path_strs = glob.glob(os.path.join(str(input_folder_path), '*.flac'))
        all_mixtures_paths = [Path(p) for p in all_mixtures_path_strs]
        
        total_tracks = len(all_mixtures_paths)

        if total_tracks == 0:
            tqdm.write("No .flac files found in the input folder.")
            tqdm.write(f"--- Finished processing folder: {input_folder_path.name} ---")
            return
            
        tqdm.write(f"Found {total_tracks} '.flac' tracks to process in {input_folder_path.name}.")

        skipped_count = 0
        attempted_processing_count = 0

        for audio_file_path in tqdm(all_mixtures_paths, desc=f"Folder: {input_folder_path.name}", unit="file", leave=False):
            file_name_base = audio_file_path.stem
            expected_output_filename = f"{file_name_base}_vocals.{self.output_format}"
            # self.output_dir is the specific output directory for the current folder being processed
            expected_output_path = self.output_dir / expected_output_filename

            if SKIP_IF_OUTPUT_EXISTS and expected_output_path.exists():
                tqdm.write(f"--> SKIPPING (output exists): {expected_output_filename} in {self.output_dir.name}")
                skipped_count += 1
                continue
            
            attempted_processing_count += 1
            self._process_file(str(audio_file_path)) # _process_file now uses tqdm.write for its logs

        elapsed_time_folder = time.time() - start_time_folder
        tqdm.write(f"\n--- Folder '{input_folder_path.name}' processing summary ---")
        tqdm.write(f"  Elapsed time: {elapsed_time_folder:.2f} seconds.")
        tqdm.write(f"  Total .flac files found: {total_tracks}")
        tqdm.write(f"  Files skipped (output already existed): {skipped_count}")
        tqdm.write(f"  Files for which processing was attempted: {attempted_processing_count}")
        tqdm.write(f"--- Finished processing folder: {input_folder_path.name} ---")


def process_batch_directories():
    BASE_INPUT_DIR = Path("/home/taresh/Downloads/anime/audios")
    BASE_OUTPUT_DIR = Path("/home/taresh/Downloads/anime/audios")

    SUBFOLDER_NAMES_TO_PROCESS: list[str] = [
        # "fate_1", 
        # "fate_2", 
        "konosuba_preq",
        "sao_1", 
        # "Rezero_s1", "Rezero_s2", "Rezero_s3p1", "Rezero_s3p2", 
        "sao_2"
    ]

    USE_GPU = True
    GPU_IDS = [0]

    print("--- Batch Processing Configuration ---")
    print(f"SKIP_IF_OUTPUT_EXISTS: {SKIP_IF_OUTPUT_EXISTS}")
    print(f"Base Input Directory: {BASE_INPUT_DIR}")
    print(f"Base Output Directory: {BASE_OUTPUT_DIR}")
    print(f"Subfolders to Process: {SUBFOLDER_NAMES_TO_PROCESS}")
    print(f"Using GPU: {USE_GPU}, IDs: {GPU_IDS if USE_GPU else 'N/A'}")
    print("-" * 40)

    processed_folders_count = 0
    skipped_folders_count = 0

    try:
        separator = VocalSeparator(
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            output_dir=str(BASE_OUTPUT_DIR),
            model_type=MODEL_TYPE,
            use_gpu=USE_GPU,
            gpu_ids=GPU_IDS,
            output_format=OUTPUT_FORMAT,
            output_pcm_type=OUTPUT_PCM_TYPE
        )
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to initialize VocalSeparator: {e}")
        print("Batch processing cannot continue.")
        return

    overall_start_time = time.time()

    for subfolder_name in tqdm(SUBFOLDER_NAMES_TO_PROCESS, desc="Overall Batch Progress", unit="folder"):
        current_input_folder = BASE_INPUT_DIR / subfolder_name / "orig"
        current_output_folder_for_vocals = BASE_OUTPUT_DIR / subfolder_name / "vocals"

        if not current_input_folder.is_dir():
            tqdm.write(f"⚠️ WARNING: Input folder '{current_input_folder}' does not exist. Skipping subfolder '{subfolder_name}'.")
            skipped_folders_count += 1
            tqdm.write("-" * 30)
            continue

        current_output_folder_for_vocals.mkdir(parents=True, exist_ok=True)

        separator.output_dir = current_output_folder_for_vocals

        try:
            separator.process_folder(str(current_input_folder))
            processed_folders_count += 1
        except Exception as e:
            tqdm.write(f"❌ ERROR processing subfolder '{subfolder_name}' (Input: {current_input_folder}): {e}")

    overall_elapsed_time = time.time() - overall_start_time
    print("\n--- Overall Batch Processing Summary ---")
    print(f"Total time for batch: {overall_elapsed_time:.2f} seconds.")
    print(f"Subfolders for which processing was initiated: {processed_folders_count}")
    print(f"Subfolders skipped (input directory not found): {skipped_folders_count}")
    print(f"Total subfolders in list: {len(SUBFOLDER_NAMES_TO_PROCESS)}")
    print("Batch processing completed!")


if __name__ == "__main__":
    process_batch_directories()