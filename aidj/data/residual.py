"""Mix residual computation.

Computes the difference between DJ mix stems and original track stems
to capture what the DJ added (extra beats, FX, loops) beyond
volume/EQ changes.
"""
from __future__ import annotations

import logging

import numpy as np
import librosa

from aidj import config

log = logging.getLogger(__name__)


def compute_residual(
    mix_stems: dict[str, np.ndarray],
    track_stems: dict[str, np.ndarray],
    sr: int = config.SR,
    n_mels: int = 128,
    n_fft: int = config.OPT_FFT,
    hop_length: int = config.OPT_HOP,
) -> dict[str, np.ndarray]:
    """Compute per-stem residual spectrograms between mix and track.

    For each stem:
        residual_audio = mix_stem - length-matched track_stem
        residual_spec  = mel_spectrogram(residual_audio)

    Args:
        mix_stems: dict stem_name -> 1D audio from the mix segment
        track_stems: dict stem_name -> 1D audio from the original track
            (already time-aligned to the mix segment)
        sr: sample rate
        n_mels: number of mel bins
        n_fft: FFT size
        hop_length: hop size

    Returns:
        dict mapping stem_name -> (n_mels, n_frames) normalized mel spectrogram
        of the residual signal
    """
    residuals = {}
    for stem in config.STEMS:
        mix_audio = mix_stems.get(stem)
        track_audio = track_stems.get(stem)

        if mix_audio is None or track_audio is None:
            continue

        # Match lengths
        min_len = min(len(mix_audio), len(track_audio))
        if min_len < sr:  # skip if less than 1 second
            continue

        mix_seg = mix_audio[:min_len].astype(np.float32)
        trk_seg = track_audio[:min_len].astype(np.float32)

        # Compute residual
        residual = mix_seg - trk_seg

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=residual, sr=sr, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels, power=1,
        )
        mel_db = librosa.amplitude_to_db(mel, ref=np.max)
        mel_db = (mel_db + 80) / 80  # normalize to ~[0, 1]

        residuals[stem] = mel_db.astype(np.float32)

    return residuals


def align_track_to_mix_segment(
    track_audio: np.ndarray,
    track_start_time: float,
    mix_segment_len: int,
    sr: int = config.SR,
) -> np.ndarray:
    """Extract and length-match a track segment to a mix segment.

    Args:
        track_audio: full track audio (1D)
        track_start_time: time in the track corresponding to the
            start of the mix segment (seconds)
        mix_segment_len: length of the mix segment in samples
        sr: sample rate

    Returns:
        Track audio segment matched to mix_segment_len
    """
    start_sample = int(track_start_time * sr)
    end_sample = start_sample + mix_segment_len
    segment = track_audio[start_sample:end_sample]

    # Pad if needed
    if len(segment) < mix_segment_len:
        segment = np.pad(segment, (0, mix_segment_len - len(segment)))

    return segment.astype(np.float32)
