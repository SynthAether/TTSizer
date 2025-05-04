
from vocal_remover import VocalSeparator


# --- Configuration ---
MODEL_PATH = "weights/kimmel_unwa_ft2_bleedless.ckpt"  # REQUIRED: Update this path
CONFIG_PATH = "configs/config_kimmel_unwa_ft.yaml" # REQUIRED: Update this path
INPUT_FOLDER = "temp/input" # REQUIRED: Update this path
OUTPUT_FOLDER = "temp/vocals" # REQUIRED: Update this path

USE_GPU = True 
GPU_IDS = [0] 
                
OUTPUT_FORMAT = 'wav' 
OUTPUT_PCM_TYPE = 'FLOAT'
MODEL_TYPE = 'mel_band_roformer' 


separator = VocalSeparator(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    output_dir=OUTPUT_FOLDER,
    model_type=MODEL_TYPE,
    use_gpu=USE_GPU,
    gpu_ids=GPU_IDS, # Pass the GPU IDs
    output_format=OUTPUT_FORMAT,
    output_pcm_type=OUTPUT_PCM_TYPE
)

def main():

    separator.process_folder(INPUT_FOLDER)

    print("\nProcessing completed successfully!")
    
    # do the gemini call for speaker diarization and get timestamps and speaker labels
    
    # desired_speaker = [""]
    
    


if __name__=="__main__":
    main()