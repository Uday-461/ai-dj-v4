"""Transition region extraction from alignment results.

Ported from djmix-dataset/djmix/alignment/transitions.py.
Identifies overlap regions between consecutive tracks in a mix based on
their DTW alignment warp paths.
"""
from __future__ import annotations

import os
import pickle
import logging
from pathlib import Path

import numpy as np

from aidj import config

log = logging.getLogger(__name__)


def extract_transitions(mix, alignment_results, data_root=None, cache_dir=None):
    """Extract transition regions from alignment results.

    For each pair of consecutive tracks, identifies the overlap region
    and computes transition metadata (overlap length, cue points, etc.).

    Args:
        mix: dict with 'id' and 'tracklist'.
        alignment_results: list of alignment result dicts from aligner.
        data_root: root data directory.
        cache_dir: cache directory for results.

    Returns:
        list of transition dicts.
    """
    data_root = Path(data_root or config.DATA_ROOT)
    cache_dir = Path(cache_dir or data_root / "results" / "transitions")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{mix['id']}.pkl"
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Build lookup: track_id -> best alignment result (by match_rate)
    best_align = {}
    for r in alignment_results:
        tid = r['track_id']
        if tid not in best_align or r['match_rate'] > best_align[tid]['match_rate']:
            best_align[tid] = r

    # Build tracklist with indices
    tracklist = mix.get('tracklist', [])
    transitions = []

    for i in range(len(tracklist) - 1):
        prev_track = tracklist[i]
        next_track = tracklist[i + 1]

        prev_id = prev_track.get('id')
        next_id = next_track.get('id')

        if prev_id is None or next_id is None:
            continue

        prev_align = best_align.get(prev_id)
        next_align = best_align.get(next_id)

        if prev_align is None or next_align is None:
            continue

        if prev_align.get('wp') is None or next_align.get('wp') is None:
            continue

        # Sanity check: track_cue_in_time must not exceed track_cue_out_time.
        # A bad DTW warping path can produce a cue_in_beat index that's larger
        # than the beat array, causing the clamped lookup to return the last beat
        # (track end), or — if the mix beat array is accidentally used — a time
        # in the hundreds of seconds. Either way the transition is unusable.
        prev_cue_in  = prev_align.get('track_cue_in_time')
        prev_cue_out = prev_align.get('track_cue_out_time')
        next_cue_in  = next_align.get('track_cue_in_time')
        next_cue_out = next_align.get('track_cue_out_time')
        MAX_TRACK_SECS = 3600  # no DJ track is longer than 1 hour
        for label, val in [('prev track_cue_in', prev_cue_in),
                           ('prev track_cue_out', prev_cue_out),
                           ('next track_cue_in', next_cue_in),
                           ('next track_cue_out', next_cue_out)]:
            if val is not None and val > MAX_TRACK_SECS:
                log.warning(
                    f"Mix {mix['id']} transition {i}: bogus cue time "
                    f"({label}={val:.0f}s > {MAX_TRACK_SECS}s), skipping"
                )
                break
        else:
            # All cue times valid — fall through to append
            pass
        if any(v is not None and v > MAX_TRACK_SECS
               for v in [prev_cue_in, prev_cue_out, next_cue_in, next_cue_out]):
            continue

        wp_prev = prev_align['wp']
        wp_next = next_align['wp']
        wp_raw_prev = prev_align['wp_raw']
        wp_raw_next = next_align['wp_raw']

        # Compute transition metrics
        last_wpt_prev = int(wp_raw_prev[0, 0])
        last_wpt_next = int(wp_raw_next[0, 0])
        total_wpt_prev = last_wpt_prev + 1
        total_wpt_next = last_wpt_next + 1

        track_cue_out_wpt_prev = int(wp_prev[0, 0])
        extra_wpts_prev = total_wpt_prev - track_cue_out_wpt_prev
        extra_wpts_next = int(wp_next[-1, 0])

        mix_cue_out_wpt_prev = int(wp_prev[0, 1])
        mix_cue_in_wpt_next = int(wp_next[-1, 1])
        tran_wpts = abs(mix_cue_out_wpt_prev - mix_cue_in_wpt_next)

        if mix_cue_out_wpt_prev > mix_cue_in_wpt_next:
            overlap_wpts = tran_wpts + extra_wpts_prev + extra_wpts_next
        else:
            overlap_wpts = (
                (mix_cue_out_wpt_prev + extra_wpts_prev)
                - (mix_cue_in_wpt_next - extra_wpts_next)
            )

        transition = {
            'tran_id': f"{mix['id']}-{i:02d}",
            'mix_id': mix['id'],
            'i_tran': i,
            'track_id_prev': prev_id,
            'track_id_next': next_id,
            'i_track_prev': i,
            'i_track_next': i + 1,

            'match_rate_prev': prev_align['match_rate'],
            'match_rate_next': next_align['match_rate'],
            'matched_beats_prev': prev_align['matched_beats'],
            'matched_beats_next': next_align['matched_beats'],

            'overlap_wpts': overlap_wpts,
            'overlap_beats': overlap_wpts / 2,
            'tran_wpts': tran_wpts,
            'extra_wpts_prev': extra_wpts_prev,
            'extra_wpts_next': extra_wpts_next,

            'total_wpt_prev': total_wpt_prev,
            'total_wpt_next': total_wpt_next,

            'wp_prev': wp_prev,
            'wp_next': wp_next,
            'wp_raw_prev': wp_raw_prev,
            'wp_raw_next': wp_raw_next,

            # Alignment metadata
            'mix_cue_in_time_prev': prev_align.get('mix_cue_in_time'),
            'mix_cue_out_time_prev': prev_align.get('mix_cue_out_time'),
            'track_cue_in_time_prev': prev_align.get('track_cue_in_time'),
            'track_cue_out_time_prev': prev_align.get('track_cue_out_time'),
            'mix_cue_in_time_next': next_align.get('mix_cue_in_time'),
            'mix_cue_out_time_next': next_align.get('mix_cue_out_time'),
            'track_cue_in_time_next': next_align.get('track_cue_in_time'),
            'track_cue_out_time_next': next_align.get('track_cue_out_time'),
        }
        transitions.append(transition)

    # Cache results
    with open(cache_path, 'wb') as f:
        pickle.dump(transitions, f)

    log.info(f"Extracted {len(transitions)} transitions from mix {mix['id']}")
    return transitions
