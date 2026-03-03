"""DTW-based track-to-mix alignment.

Ported from djmix-dataset/djmix/alignment/dtw.py.
Uses cosine distance on concatenated (chroma + MFCC) halfbeat features,
then librosa subsequence DTW to find where each track appears in the mix.
"""
from __future__ import annotations

import os
import pickle
import logging

import numpy as np
import librosa
from scipy.spatial.distance import cdist

from aidj import config
from aidj.data.helpers import correct_wp, drop_weird_wp_segments

log = logging.getLogger(__name__)

NUM_FORGIVABLE_WARP_POINTS = 2
WARP_POINTS_PER_BEAT = 2
MIN_WP_BEATS = 16


def align_track_to_mix(
    track_chroma: np.ndarray,
    track_mfcc: np.ndarray,
    track_beats: np.ndarray,
    mix_chroma: np.ndarray,
    mix_mfcc: np.ndarray,
    mix_beats: np.ndarray,
    mix_id: str,
    track_id: str,
) -> dict | None:
    """Align a single track to a mix using DTW on halfbeat features.

    Args:
        track_chroma: (12, n_halfbeats) halfbeat chroma features for track.
        track_mfcc: (20, n_halfbeats) halfbeat MFCC features for track.
        track_beats: corrected beat times for track.
        mix_chroma: (12, n_halfbeats) halfbeat chroma features for mix.
        mix_mfcc: (20, n_halfbeats) halfbeat MFCC features for mix.
        mix_beats: corrected beat times for mix.
        mix_id: mix identifier.
        track_id: track identifier.

    Returns:
        dict with alignment results, or None if alignment fails.
    """
    if len(track_beats) < 16:
        log.debug(f"Track {track_id} has too few beats ({len(track_beats)}), skipping")
        return None

    # Z-score normalization using track statistics
    track_mfcc_means = track_mfcc.mean(axis=1, keepdims=True)
    track_mfcc_stds = track_mfcc.std(axis=1, keepdims=True)
    track_mfcc_scaled = (track_mfcc - track_mfcc_means) / (track_mfcc_stds + 1e-10)
    mix_mfcc_scaled = (mix_mfcc - track_mfcc_means) / (track_mfcc_stds + 1e-10)
    track_mfcc_scaled = np.nan_to_num(track_mfcc_scaled)
    mix_mfcc_scaled = np.nan_to_num(mix_mfcc_scaled)

    track_chroma_means = track_chroma.mean(axis=1, keepdims=True)
    track_chroma_stds = track_chroma.std(axis=1, keepdims=True)
    track_chroma_scaled = (track_chroma - track_chroma_means) / (track_chroma_stds + 1e-10)
    mix_chroma_scaled = (mix_chroma - track_chroma_means) / (track_chroma_stds + 1e-10)
    track_chroma_scaled = np.nan_to_num(track_chroma_scaled)
    mix_chroma_scaled = np.nan_to_num(mix_chroma_scaled)

    # Concatenate features
    track_all_scaled = np.concatenate([track_chroma_scaled, track_mfcc_scaled])
    mix_all_scaled = np.concatenate([mix_chroma_scaled, mix_mfcc_scaled])

    # Compute cost matrix and DTW
    C = cdist(track_all_scaled.T, mix_all_scaled.T, metric='cosine')
    C = np.nan_to_num(C)
    D, wp_raw = librosa.sequence.dtw(C=C, subseq=True)

    # Matching function and cost
    matching_function = D[-1, :] / wp_raw.shape[0]
    cost = float(matching_function.min())

    # Correct warp points
    wp = correct_wp(wp_raw, NUM_FORGIVABLE_WARP_POINTS)

    if wp is not None and wp.size > 0:
        wp = drop_weird_wp_segments(
            wp, NUM_FORGIVABLE_WARP_POINTS, MIN_WP_BEATS, WARP_POINTS_PER_BEAT,
        )

    # Compute match rates
    total_beats = len(track_beats)
    wp_mix_raw = wp_raw[:, 1][::-1]
    wp_trk_raw = wp_raw[:, 0][::-1]
    dydx_raw = np.diff(wp_trk_raw) / np.maximum(np.diff(wp_mix_raw), 1e-10)
    matched_beats_raw = int((np.abs(dydx_raw - 1) < 0.01).sum() // WARP_POINTS_PER_BEAT)
    match_rate_raw = matched_beats_raw / max(total_beats, 1)

    result = {
        'mix_id': mix_id,
        'track_id': track_id,
        'cost': cost,
        'wp_raw': wp_raw,
        'match_rate_raw': match_rate_raw,
        'matched_beats_raw': matched_beats_raw,
    }

    if wp is not None and wp.size > 0:
        wp_mix = wp[:, 1][::-1]
        wp_trk = wp[:, 0][::-1]
        matched_beats = int((wp_trk[-1] - wp_trk[0] + 1) // WARP_POINTS_PER_BEAT)
        match_rate = matched_beats / max(total_beats, 1)

        mix_cue_in_beat = int(wp_mix[0] // 2)
        mix_cue_out_beat = int(wp_mix[-1] // 2)
        track_cue_in_beat = int(wp_trk[0] // 2)
        track_cue_out_beat = int(wp_trk[-1] // 2)

        # Safely index beat times
        mix_cue_in_time = float(mix_beats[min(mix_cue_in_beat, len(mix_beats) - 1)])
        mix_cue_out_time = float(mix_beats[min(mix_cue_out_beat, len(mix_beats) - 1)])
        track_cue_in_time = float(track_beats[min(track_cue_in_beat, len(track_beats) - 1)])
        track_cue_out_time = float(track_beats[min(track_cue_out_beat, len(track_beats) - 1)])

        result.update({
            'wp': wp,
            'match_rate': match_rate,
            'matched_beats': matched_beats,
            'mix_cue_in_beat': mix_cue_in_beat,
            'mix_cue_out_beat': mix_cue_out_beat,
            'track_cue_in_beat': track_cue_in_beat,
            'track_cue_out_beat': track_cue_out_beat,
            'mix_cue_in_time': mix_cue_in_time,
            'mix_cue_out_time': mix_cue_out_time,
            'track_cue_in_time': track_cue_in_time,
            'track_cue_out_time': track_cue_out_time,
        })
    else:
        result.update({
            'wp': None,
            'match_rate': 0.0,
            'matched_beats': 0,
            'mix_cue_in_beat': None,
            'mix_cue_out_beat': None,
            'track_cue_in_beat': None,
            'track_cue_out_beat': None,
            'mix_cue_in_time': None,
            'mix_cue_out_time': None,
            'track_cue_in_time': None,
            'track_cue_out_time': None,
        })

    return result


def align_mix(mix, data_root=None, cache_dir=None):
    """Align all tracks in a mix.

    Loads (or computes) beat and feature data for the mix and each track,
    then runs ``align_track_to_mix`` for every track in the tracklist.
    Results are cached as pickle files.

    Args:
        mix: dict with 'id' and 'tracklist' fields.
        data_root: root data directory (defaults to ``config.DATA_ROOT``).
        cache_dir: directory for cached alignment results.

    Returns:
        list of alignment result dicts.
    """
    import gc
    from pathlib import Path
    from aidj.data.beat_detector import (
        BeatDetector, compute_halfbeat_chroma, compute_halfbeat_mfcc,
    )

    data_root = Path(data_root or config.DATA_ROOT)
    cache_dir = Path(cache_dir or data_root / "results" / "alignments")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{mix['id']}.pkl"
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    mix_path = str(data_root / "mixes" / f"{mix['id']}.mp3")
    if not os.path.isfile(mix_path):
        log.warning(f"Mix audio not found: {mix_path}")
        return []

    # Load mix beats and features
    beats_dir = data_root / "beats"
    mix_beats_path = beats_dir / f"{mix['id']}.npz"
    if mix_beats_path.exists():
        data = np.load(mix_beats_path)
        mix_beats = data['beats']
    else:
        detector = BeatDetector(method="librosa")
        mix_beats, _ = detector.corrected_beats(mix_path)

    if len(mix_beats) < 16:
        log.warning(f"Mix {mix['id']} has too few beats")
        return []

    # Load mix audio once at FEATURE_SR (22050Hz — sufficient for chroma/MFCC,
    # half the RAM of 44100Hz), compute both features, then free before track loop
    mix_y, _ = librosa.load(mix_path, sr=config.FEATURE_SR, mono=True)
    mix_chroma = compute_halfbeat_chroma(mix_path, mix_beats, y=mix_y)
    mix_mfcc = compute_halfbeat_mfcc(mix_path, mix_beats, y=mix_y)
    del mix_y
    gc.collect()

    results = []
    for track in mix.get('tracklist', []):
        track_id = track.get('id')
        if track_id is None:
            continue

        track_path = str(data_root / "tracks" / f"{track_id}.mp3")
        if not os.path.isfile(track_path):
            continue

        try:
            # Load track beats
            track_beats_path = beats_dir / f"{track_id}.npz"
            if track_beats_path.exists():
                data = np.load(track_beats_path)
                track_beats = data['beats']
            else:
                detector = BeatDetector(method="librosa")
                track_beats, _ = detector.corrected_beats(track_path)

            if len(track_beats) < 16:
                continue

            track_y, _ = librosa.load(track_path, sr=config.FEATURE_SR, mono=True)
            track_chroma = compute_halfbeat_chroma(track_path, track_beats, y=track_y)
            track_mfcc = compute_halfbeat_mfcc(track_path, track_beats, y=track_y)
            del track_y
            gc.collect()

            result = align_track_to_mix(
                track_chroma, track_mfcc, track_beats,
                mix_chroma, mix_mfcc, mix_beats,
                mix['id'], track_id,
            )
            if result is not None:
                results.append(result)
                log.info(
                    f"Aligned {track_id} to {mix['id']}: "
                    f"match_rate={result['match_rate']:.3f}"
                )
        except Exception as e:
            log.warning(f"Alignment failed for {track_id}: {e}")

    # Cache results
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)

    return results
