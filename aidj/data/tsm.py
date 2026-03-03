"""Time-scale modification for beat-aligned audio.

Ported from djmix-dataset/djmix/alignment/tsm.py.
Uses pytsmod WSOLA for high-quality time stretching guided by DTW anchor points.
"""
from __future__ import annotations

import numpy as np
import librosa
from numpy.typing import ArrayLike

from aidj.data.helpers import extend_wp, anchors


def halfbeats(beats):
    """Compute halfbeat times from beat times.

    Inserts a point halfway between each consecutive pair of beats.

    Args:
        beats: 1D array of beat times in seconds.

    Returns:
        1D array of halfbeat times (length = 2 * len(beats) - 1).
    """
    hb = np.zeros(len(beats) * 2 - 1)
    hb[0::2] = beats
    hb[1::2] = beats[:-1] + (np.diff(beats) / 2)
    return hb


def _rms(x):
    """Root mean square of an array."""
    return np.sqrt(np.mean(np.abs(x) ** 2))


def aligned_tsm(
    wp_prev: ArrayLike,
    wp_next: ArrayLike,
    total_wpt_prev: int,
    total_wpt_next: int,
    mix: ArrayLike,
    track_prev: ArrayLike,
    track_next: ArrayLike,
    beats_mix: ArrayLike,
    beats_prev: ArrayLike,
    beats_next: ArrayLike,
    pad_beats: int,
    sr: int,
    gain_normalization_sec: int = 5,
):
    """Perform time-scale modification using WSOLA to align tracks to mix timing.

    Given warp paths for two consecutive tracks (previous and next), this
    function stretches/compresses the track audio so that their beats line
    up with the corresponding mix beats in the transition region.

    Args:
        wp_prev: warp path for the previous track (N, 2).
        wp_next: warp path for the next track (M, 2).
        total_wpt_prev: total warp points in previous track.
        total_wpt_next: total warp points in next track.
        mix: mix audio array, shape (channels, samples).
        track_prev: previous track audio, shape (channels, samples).
        track_next: next track audio, shape (channels, samples).
        beats_mix: beat times for the mix.
        beats_prev: beat times for the previous track.
        beats_next: beat times for the next track.
        pad_beats: extra beats of context to include.
        sr: sample rate.
        gain_normalization_sec: seconds of audio to use for gain normalization.
            Set to 0 or None to disable.

    Returns:
        (tran_mix, tran_prev, tran_next) -- the transition region of the mix
        and the time-stretched track segments, all with the same length.
    """
    import pytsmod as tsm

    halfbeats_mix = halfbeats(beats_mix)
    halfbeats_prev = halfbeats(beats_prev)
    halfbeats_next = halfbeats(beats_next)

    wp_prev = extend_wp(wp_prev, total_wpt_prev)
    wp_next = extend_wp(wp_next, total_wpt_next)

    # Warping points on the mix.
    start_wpt_mix = wp_next[-1, 1]
    last_wpt_mix = wp_prev[0, 1]

    # Start earlier for the previous track.
    start_wpt_mix_prev = max(start_wpt_mix - (pad_beats * 2), 0)
    anchors_trk_prev, anchors_mix_prev = anchors(
        wp_prev, start_wpt_mix_prev, last_wpt_mix, halfbeats_mix, halfbeats_prev,
    )
    i_anc_trk_prev = librosa.time_to_samples(anchors_trk_prev, sr=sr)
    i_anc_mix_prev = librosa.time_to_samples(anchors_mix_prev, sr=sr)

    # End later for the next track.
    last_wpt_mix_next = min(last_wpt_mix + (pad_beats * 2), wp_next[0, 1])
    anchors_trk_next, anchors_mix_next = anchors(
        wp_next, start_wpt_mix, last_wpt_mix_next, halfbeats_mix, halfbeats_next,
    )
    i_anc_trk_next = librosa.time_to_samples(anchors_trk_next, sr=sr)
    i_anc_mix_next = librosa.time_to_samples(anchors_mix_next, sr=sr)

    # The optimizing audio segment of the mix.
    tran_mix = mix[:, i_anc_mix_prev[0]:i_anc_mix_next[-1] + 1]

    # Time scale modification for the previous track.
    tran_prev_raw = track_prev[:, i_anc_trk_prev[0]:i_anc_trk_prev[-1] + 1]
    tsm_prev = np.array([
        librosa.time_to_samples(anchors_trk_prev - anchors_trk_prev[0], sr=sr),
        librosa.time_to_samples(anchors_mix_prev - anchors_mix_prev[0], sr=sr),
    ])
    tran_prev = tsm.wsola(tran_prev_raw, tsm_prev).astype('float32')
    # Right-pad for the previous track.
    tran_prev = librosa.util.fix_length(tran_prev, size=tran_mix.shape[1])

    # Time scale modification for the next track.
    tran_next_raw = track_next[:, i_anc_trk_next[0]:i_anc_trk_next[-1] + 1]
    tsm_next = np.array([
        librosa.time_to_samples(anchors_trk_next - anchors_trk_next[0], sr=sr),
        librosa.time_to_samples(anchors_mix_next - anchors_mix_next[0], sr=sr),
    ])
    tran_next = tsm.wsola(tran_next_raw, tsm_next).astype('float32')
    # Left-pad for the next track.
    tran_next = librosa.util.fix_length(
        tran_next[:, ::-1], size=tran_mix.shape[1]
    )[:, ::-1]

    if gain_normalization_sec:
        gain_equalizing_samples = int(sr * gain_normalization_sec)
        rms_mix_start = _rms(tran_mix[:, :gain_equalizing_samples])
        rms_prev_start = _rms(tran_prev[:, :gain_equalizing_samples])
        if rms_prev_start > 0:
            gain_prev = rms_mix_start / rms_prev_start
        else:
            gain_prev = 1.0
        rms_mix_end = _rms(tran_mix[:, -gain_equalizing_samples:])
        rms_next_end = _rms(tran_next[:, -gain_equalizing_samples:])
        if rms_next_end > 0:
            gain_next = rms_mix_end / rms_next_end
        else:
            gain_next = 1.0
        tran_prev *= gain_prev
        tran_next *= gain_next

    return tran_mix, tran_prev, tran_next
