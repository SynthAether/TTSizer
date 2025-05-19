# TTSizer

TTSizer is a powerful audio processing pipeline designed for extracting, processing, and transcribing speech from video content, with a particular focus on anime and similar media. The pipeline includes advanced features like vocal isolation, speaker diarization, and high-quality transcription.

## Features

- **Audio Extraction**: Extracts audio from video files with language preference support
- **Vocal Isolation**: Uses Mel-Band Roformer model for high-quality vocal separation
- **Audio Normalization**: Normalizes audio levels for consistent quality
- **Speaker Diarization**: Leverages Gemini LLM for accurate speaker identification
- **CTC Forced Alignment**: Precise alignment of audio with transcriptions
- **Outlier Detection**: Identifies and filters out non-target speaker segments
- **ASR Transcription**: High-quality transcription using NVIDIA's Parakeet model

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ttsizer.git
cd ttsizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model weights:
- Mel-Band Roformer model: `weights/kimmel_unwa_ft2_bleedless.ckpt`
- WeSpeaker model: `weights/wespeaker-voxceleb-resnet293-LM`

## Configuration

The pipeline is configured through `ttsizer/configs/config.yaml`. Key configuration sections:

- `project_setup`: Project name and directory paths
- `pipeline_control`: Control pipeline execution flow
- Stage-specific configurations for each processing step

## Usage

1. Configure your project in `config.yaml`:
```yaml
project_setup:
  project_name: "YourProject"
  video_source_parent_dir: "/path/to/videos"
  output_base_dir: "/path/to/output"
  target_speaker_labels_for_diarization: ["Speaker1", "Speaker2"]
```

2. Run the pipeline:
```bash
python -m ttsizer.main
```

Or specify a custom config path:
```bash
TTSIZER_CONFIG=/path/to/config.yaml python -m ttsizer.main
```

## Pipeline Stages

1. **Audio Extraction** (`01_extracted_audio`)
   - Extracts audio from video files
   - Supports multiple language tracks
   - Configurable output format and quality

2. **Vocal Removal** (`02_vocals_removed`)
   - Isolates vocals using Mel-Band Roformer
   - GPU-accelerated processing
   - High-quality output format

3. **Audio Normalization** (`03_vocals_normalized`)
   - Normalizes audio levels
   - Configurable LUFS and true peak targets
   - Multi-process support

4. **LLM Diarization** (`04_llm_diarized_json`)
   - Speaker identification using Gemini
   - JSON output format
   - Configurable model parameters

5. **CTC Alignment** (`05_aligned_clips`)
   - Forced alignment of audio and text
   - Configurable padding and thresholds
   - GPU-accelerated processing

6. **Outlier Detection** (`06_outliers_filtered`)
   - Speaker verification
   - Outlier and uncertain clip handling
   - Configurable thresholds

7. **Parakeet ASR** (`07_parakeet_transcribed`)
   - High-quality transcription
   - Flagged clip handling
   - GPU-accelerated processing

## Output Structure

```
{output_base_dir}/{project_name}/
├── 01_extracted_audio/
├── 02_vocals_removed/
├── 03_vocals_normalized/
├── 04_llm_diarized_json/
├── 05_aligned_clips/
├── 06_outliers_filtered/
└── 07_parakeet_transcribed/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your License Here]

## Acknowledgments

- NVIDIA Parakeet for ASR
- Google Gemini for LLM capabilities
- WeSpeaker for speaker embedding
- Mel-Band Roformer for vocal separation