### pipeline

input_videos (mkv - video+audio) -> extract_audio (44.1 KHz, flac-pcm-24, stereo/mono) -> vocal_removal (44.1 KHz, flac-pcm-24, stereo/mono) -> normalize (44.1 KHz, flac-pcm-24, mono) -> gemini_call -> ctc_forced_aligner (44.1 KHz, flac-pcm-24, mono) -> find_outliers -> (16Khz, wav, mono) parakeet -> audio clips (44.1 Kz, mono, wav-24) + transcriptions (.txt)