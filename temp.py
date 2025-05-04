
import warnings
warnings.filterwarnings("ignore")


from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
    load_audio,
)
import torch
import copy
import soundfile as sf
import math


device = "cuda"

# audio_path = "/Workspace/tr/repos/livekit/samples/kokoro-pt.wav"
audio_path = "temp/segmented_vocals_e10_padded/Armin_Arlert/wav/Armin_Arlert_0055.wav"

def merge_close_segments(segments, threshold):
    """
    Merges adjacent segments if the time gap between them is less than
    or equal to a specified threshold. Assumes segments are sorted by time.

    Args:
        segments: A list of dictionaries, where each dictionary represents a
                  segment and must contain 'start', 'end', and 'text' keys.
                  It can optionally contain a 'score' key.
                  Example: [{'start': 0.1, 'end': 0.5, 'text': 'Hello', 'score': -2.5}, ...]
        threshold: The maximum time gap (in seconds) allowed between the end
                   of one segment and the start of the next for them to be merged.
                   A value of 0.1 means segments with <= 100ms gap will be merged.

    Returns:
        A new list of merged segment dictionaries with updated 'end' times,
        concatenated 'text', and potentially aggregated 'scores'.
    """
    if not segments:
        return []

    # Ensure segments are sorted by start time, just in case (though aligner output usually is)
    segments.sort(key=lambda x: x['start'])

    # Start with the first segment as the initial merged segment
    # Use deepcopy to ensure we don't modify the original list's dictionary
    # if only one segment exists or the first segment is never merged into.
    merged_segments = [copy.deepcopy(segments[0])]

    for i in range(1, len(segments)):
        current_segment = segments[i]
        # Get the last segment added to our *merged* list
        last_merged = merged_segments[-1]

        # Calculate the time gap between the end of the last merged segment
        # and the start of the current segment from the input list
        gap = current_segment['start'] - last_merged['end']

        if gap <= threshold:
            # --- Merge the current segment into the last merged segment ---

            # Update the end time to the end time of the current segment
            last_merged['end'] = current_segment['end']

            # Concatenate the text with a space
            last_merged['text'] += " " + current_segment['text']

            # Optional: Handle scores if present (e.g., sum them)
            # Check if both segments have scores before attempting arithmetic
            if 'score' in current_segment and 'score' in last_merged:
                # Example: Summing scores
                last_merged['score'] += current_segment['score']
            elif 'score' in current_segment and 'score' not in last_merged:
                # If the last merged segment didn't have a score initially
                 last_merged['score'] = current_segment['score']
            # If only last_merged has score, or neither has score, do nothing to score

        else:
            # --- Gap is too large, start a new merged segment ---
            # Append a deep copy of the current segment to the merged list
            merged_segments.append(copy.deepcopy(current_segment))

    return merged_segments


def crop_wav(end_time: float, input_path: str=audio_path, output_path: str='output.wav'):
    """
    Crop a WAV file from the start up to end_time (in seconds),
    preserving original sample rate, channels and subtype (no quality loss).
    """
    # 1. Read metadata
    info = sf.info(input_path)
    samplerate = info.samplerate
    subtype    = info.subtype
    channels   = info.channels

    # 2. Compute sample index to cut at
    end_sample = math.ceil(end_time * samplerate)

    # 3. Read only the portion you need
    #    always_2d ensures shape (samples, channels)
    data, sr = sf.read(input_path, start=0, stop=end_sample,
                       dtype='float64', always_2d=True)

    # 4. Write cropped audio with identical parameters
    sf.write(output_path, data, samplerate, subtype=subtype)
    

def get_timestamps():

    # full_transcript = "Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache licensed weights, Kokoro can be deployed anywhere from production environments to personal projects."

    # full_transcript = "Kokoro is an open-weight TTS model with 82 million parameters. it delivers comparable quality to larger models while being significantly faster and more cost-efficient. Kokoro can be deployed anywhere from production environments to personal projects."
    full_transcript = "You two just act as non-aggressively as you possibly can, agreed?"

    print(full_transcript)

    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16
    )

    audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)


    emissions, stride = generate_emissions(
        alignment_model,
        audio_waveform,
        batch_size=8,
    )

    print(f"emissions: {emissions.shape}, stride: {stride}")

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language='en',
        split_size='word',
        star_frequency='edges'
    )

    print(f"test_starred: {text_starred}")


    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    # print(f"sengments: {segments}")

    spans = get_spans(tokens_starred, segments, blank_token)
    # print(spans[11])

    word_timestamps = postprocess_results(text_starred=text_starred, spans=spans, stride=stride, scores=scores)
    
    print()
    print(word_timestamps)
    print("")
    
    ending_timestamp = word_timestamps[-1]['end']
    
    print(ending_timestamp)
    
    crop_wav(end_time=ending_timestamp)
    
    # word_timestamps = merge_close_segments(word_timestamps, 0.8)

    # print(f'word timestamps: {word_timestamps}')
    
    return word_timestamps




def audio_segments(segments):
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(audio_path)

    for i, seg in enumerate(segments):
        # pydub works in milliseconds
        start_ms = int(seg["start"] * 1000)
        end_ms   = int(seg["end"]   * 1000)
        
        chunk = audio[start_ms:end_ms]
        chunk.export(f"segments/{i}.wav", format="wav")
        print(f"Exported [{seg['start']}–{seg['end']}-{seg['text']}]")
    
    
    
    
if __name__=="__main__":
    

    timestamps = get_timestamps()
    # audio_segments(timestamps)



