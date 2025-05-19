import argparse
import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def normalize_audio(audio: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize an audio signal by subtracting the mean and dividing by the standard deviation.

    Parameters:
    ----------
    audio : np.ndarray
        Input audio array with shape (channels, time) or (time,).

    Returns:
    -------
    tuple[np.ndarray, dict[str, float]]
        - Normalized audio array with the same shape as the input.
        - Dictionary containing the mean and standard deviation of the original audio.
    """

    mono = audio.mean(0)
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: Dict[str, float]) -> np.ndarray:
    """
    Denormalize an audio signal by reversing the normalization process (multiplying by the standard deviation
    and adding the mean).

    Parameters:
    ----------
    audio : np.ndarray
        Normalized audio array to be denormalized.
    norm_params : dict[str, float]
        Dictionary containing the 'mean' and 'std' values used for normalization.

    Returns:
    -------
    np.ndarray
        Denormalized audio array with the same shape as the input.
    """

    return audio * norm_params["std"] + norm_params["mean"]


def apply_tta(
        config,
        model: torch.nn.Module,
        mix: torch.Tensor,
        waveforms_orig: Dict[str, torch.Tensor],
        device: torch.device,
        model_type: str
) -> Dict[str, torch.Tensor]:
    """
    Apply Test-Time Augmentation (TTA) for source separation.

    This function processes the input mixture with test-time augmentations, including
    channel inversion and polarity inversion, to enhance the separation results. The
    results from all augmentations are averaged to produce the final output.

    Parameters:
    ----------
    config : Any
        Configuration object containing model and processing parameters.
    model : torch.nn.Module
        The trained model used for source separation.
    mix : torch.Tensor
        The mixed audio tensor with shape (channels, time).
    waveforms_orig : Dict[str, torch.Tensor]
        Dictionary of original separated waveforms (before TTA) for each instrument.
    device : torch.device
        Device (CPU or CUDA) on which the model will be executed.
    model_type : str
        Type of the model being used (e.g., "demucs", "custom_model").

    Returns:
    -------
    Dict[str, torch.Tensor]
        Updated dictionary of separated waveforms after applying TTA.
    """
    # Create augmentations: channel inversion and polarity inversion
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]

    # Process each augmented mixture
    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = demix(config, model, augmented_mix, device, model_type=model_type)
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]

    # Average the results across augmentations
    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1

    return waveforms_orig


def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate a windowing array with a linear fade-in at the beginning and a fade-out at the end.

    This function creates a window of size `window_size` where the first `fade_size` elements
    linearly increase from 0 to 1 (fade-in) and the last `fade_size` elements linearly decrease
    from 1 to 0 (fade-out). The middle part of the window is filled with ones.

    Parameters:
    ----------
    window_size : int
        The total size of the window.
    fade_size : int
        The size of the fade-in and fade-out regions.

    Returns:
    -------
    torch.Tensor
        A tensor of shape (window_size,) containing the generated windowing array.

    Example:
    -------
    If `window_size=10` and `fade_size=3`, the output will be:
    tensor([0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0.0000])
    """

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def demix(
        config: ConfigDict,
        model: torch.nn.Module,
        mix: torch.Tensor,
        device: torch.device,
        model_type: str,
        pbar: bool = False
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """
    Unified function for audio source separation with support for multiple processing modes.

    This function separates audio into its constituent sources using either a generic custom logic
    or a Demucs-specific logic. It supports batch processing and overlapping window-based chunking
    for efficient and artifact-free separation.

    Parameters:
    ----------
    config : ConfigDict
        Configuration object containing audio and inference settings.
    model : torch.nn.Module
        The trained model used for audio source separation.
    mix : torch.Tensor
        Input audio tensor with shape (channels, time).
    device : torch.device
        The computation device (CPU or CUDA).
    model_type : str, optional
        Processing mode:
            - "demucs" for logic specific to the Demucs model.
        Default is "generic".
    pbar : bool, optional
        If True, displays a progress bar during chunk processing. Default is False.

    Returns:
    -------
    Union[Dict[str, np.ndarray], np.ndarray]
        - A dictionary mapping target instruments to separated audio sources if multiple instruments are present.
        - A numpy array of the separated source if only one instrument is present.
    """

    mix = torch.tensor(mix, dtype=torch.float32)

    if model_type == 'htdemucs':
        mode = 'demucs'
    else:
        mode = 'generic'
    # Define processing parameters based on the mode
    if mode == 'demucs':
        chunk_size = config.training.samplerate * config.training.segment
        num_instruments = len(config.training.instruments)
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        chunk_size = config.audio.chunk_size
        num_instruments = len([config.training.target_instrument])
        num_overlap = config.inference.num_overlap

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = _getWindowingArray(chunk_size, fade_size)
        # Add padding for generic mode to handle edge artifacts
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size

    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # Initialize result and counter tensors
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(
                total=mix.shape[1], desc="Processing audio chunks", leave=False
            ) if pbar else None

            while i < mix.shape[1]:
                # Extract chunk and apply padding if necessary
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                if mode == "generic" and chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                # Process batch if it's full or the end is reached
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    if mode == "generic":
                        window = windowing_array.clone() # using clone() fixes the clicks at chunk edges when using batch_size=1
                        if i - step == 0:  # First audio chunk, no fadein
                            window[:fade_size] = 1
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window[-fade_size:] = 1

                    for j, (start, seg_len) in enumerate(batch_locations):
                        if mode == "generic":
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                            counter[..., start:start + seg_len] += window[..., :seg_len]
                        else:
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu()
                            counter[..., start:start + seg_len] += 1.0

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            # Compute final estimated sources
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            # Remove padding for generic mode
            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    # Return the result as a dictionary or a single array
    if mode == "demucs":
        instruments = config.training.instruments
    else:
        instruments = [config.training.target_instrument]

    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}

    if mode == "demucs" and num_instruments <= 1:
        return estimated_sources
    else:
        return ret_data


def load_start_checkpoint(args: argparse.Namespace, model: torch.nn.Module, type_='train') -> None:
    """
    Load the starting checkpoint for a model.

    Args:
        args: Parsed command-line arguments containing the checkpoint path.
        model: PyTorch model to load the checkpoint into.
        type_: how to load weights - for train we can load not fully compatible weights
    """

    print(f'Start from checkpoint: {args.start_check_point}')
    device='cpu'
    state_dict = torch.load(args.start_check_point, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
