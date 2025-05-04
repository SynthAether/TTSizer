import nemo.collections.asr as nemo_asr
import soundfile as sf
import librosa

audio_file = "output.wav"
out_path = "output_16khz_mono.wav"

def main():
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    output = asr_model.transcribe([out_path])
    print(output[0].text)

def convert_to_mono_16k_librosa():
    # librosa.load returns mono by default and resamples if sr is given
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    sf.write(out_path, y, 16000)

if __name__=="__main__":
    # convert_to_mono_16k_librosa()
    
    main()
    