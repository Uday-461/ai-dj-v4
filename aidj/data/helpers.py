"""Warp-path helper utilities for DTW alignment.

Ported from djmix-dataset/djmix/alignment/helpers.py with pandas dependency
removed (uses numpy equivalents instead).
"""
from __future__ import annotations

import numpy as np


def diff(a, n=1):
    """Simple array difference: a[n:] - a[:-n]."""
    return a[n:] - a[:-n]


def correct_wp(wp, num_forgivable_points=2):
    """Select consecutive warp points, forgiving small gaps.

    Identifies warp-path points where both mix and track coordinates
    advance over a sliding window of size ``num_forgivable_points + 1``.
    Duplicate x or y coordinates are then removed.

    Args:
        wp: (N, 2) warp path array, columns are (track_idx, mix_idx),
            ordered from high to low indices (as returned by librosa DTW).
        num_forgivable_points: tolerance for non-advancing segments.

    Returns:
        Corrected warp path array, or None if the result is empty.
    """
    if wp is None or wp.size == 0:
        return None

    diff_n = num_forgivable_points + 1

    wp_mix_fwd = wp[:, 1][::-1]
    wp_trk_fwd = wp[:, 0][::-1]
    wp_mix_bwd = wp[:, 1]
    wp_trk_bwd = wp[:, 0]

    dmix_fwd = diff(wp_mix_fwd, diff_n)
    dtrk_fwd = diff(wp_trk_fwd, diff_n)
    dmix_bwd = diff(wp_mix_bwd, diff_n)
    dtrk_bwd = diff(wp_trk_bwd, diff_n)

    crt_fwd = np.logical_and(dmix_fwd, dtrk_fwd)
    crt_bwd = np.logical_and(dmix_bwd, dtrk_bwd)[::-1]
    crt_both = np.logical_and(
        np.concatenate([crt_fwd, [True] * diff_n]),
        np.concatenate([[True] * diff_n, crt_bwd]),
    )
    wp_crt = wp[crt_both[::-1]]

    if wp_crt.size == 0:
        return None

    # Drop points where x or y coordinate is duplicated.
    # Equivalent to pd.Series().duplicated() -- marks True for second+ occurrences.
    _, unique_idx_x = np.unique(wp_crt[:, 0], return_index=True)
    _, unique_idx_y = np.unique(wp_crt[:, 1], return_index=True)
    dup_x = np.ones(len(wp_crt), dtype=bool)
    dup_x[unique_idx_x] = False
    dup_y = np.ones(len(wp_crt), dtype=bool)
    dup_y[unique_idx_y] = False
    duplicated = dup_x | dup_y
    wp_crt_unique = wp_crt[~duplicated]

    if wp_crt_unique.size == 0:
        return None

    return wp_crt_unique


def drop_weird_wp_segments(wp, num_forgivable_points=2, min_wp_beats=16,
                           warp_points_per_beat=2):
    """Group adjacent warp paths and select the longest consistent group.

    Splits the warp path at points where consecutive rows jump by more than
    ``num_forgivable_points``.  Keeps only segments with at least
    ``min_wp_beats * warp_points_per_beat`` points, then selects those whose
    median intercept is close to the longest segment's intercept.

    Args:
        wp: corrected warp path array (N, 2).
        num_forgivable_points: max average jump before splitting.
        min_wp_beats: minimum number of beats for a valid segment.
        warp_points_per_beat: warp points per beat (typically 2 for halfbeats).

    Returns:
        Filtered warp path array, or None if nothing survives.
    """
    if wp is None or wp.size == 0:
        return None

    wp_diff = np.abs(np.concatenate([wp[1:], [wp[-1]]]) - wp).mean(axis=1)
    borders = np.flatnonzero(wp_diff > num_forgivable_points) + 1
    wp_groups = np.split(wp, borders)
    min_warp_points = min_wp_beats * warp_points_per_beat
    wp_long_enough = [wg for wg in wp_groups if len(wg) > min_warp_points]

    if len(wp_long_enough) == 0:
        return None

    # Compute length, perform linear regression without outliers
    # to get intercepts (b from y = ax + b).
    wp_lengths, wp_intercepts = [], []
    for wp_seg in wp_long_enough:
        y, x = wp_seg[::-1].T
        dx, dy = np.diff(x), np.diff(y)
        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = dy / dx
        intercepts = y[:-1] - slopes * x[:-1]
        median_intercept = np.median(intercepts[np.isfinite(intercepts)])
        wp_lengths.append(len(wp_seg))
        wp_intercepts.append(median_intercept)

    # Select warp path segments on the same line as the longest segment.
    wp_longest_intercept = wp_intercepts[np.argmax(wp_lengths)]
    wp_verified = [
        wp_seg for wp_seg, b in zip(wp_long_enough, wp_intercepts)
        if abs(b - wp_longest_intercept) <= num_forgivable_points
    ]

    if len(wp_verified) == 0:
        return None

    wp_verified = np.concatenate(wp_verified)
    return wp_verified


def find_cue(wp, cue_in=False, num_diag=32):
    """Find cue points from a warp path.

    Identifies the point where ``num_diag`` consecutive diagonal (slope-1)
    steps begin (cue-in) or end (cue-out).

    Args:
        wp: warp path array (N, 2), descending order.
        cue_in: if True, return cue-in point; otherwise cue-out.
        num_diag: number of diagonal steps to require.

    Returns:
        (cue_mix, cue_track) tuple of integer warp-point indices.
    """
    if num_diag == 0:
        if cue_in:
            return wp[-1, 1], wp[-1, 0]
        else:
            return wp[0, 1], wp[0, 0]

    x, y = wp[::-1, 1], wp[::-1, 0]
    dx, dy = np.diff(x), np.diff(y)

    with np.errstate(divide='ignore'):
        slope = dy / dx
    slope[np.isinf(slope)] = 0

    if cue_in:
        slope = slope[::-1].cumsum()
        slope[num_diag:] = slope[num_diag:] - slope[:-num_diag]
        slope = slope[::-1]
        i_diag = np.nonzero(slope == num_diag)[0]
        if len(i_diag) == 0:
            return find_cue(wp, cue_in, num_diag // 2)
        else:
            i = i_diag[0]
            return x[i], y[i]
    else:
        slope = slope.cumsum()
        slope[num_diag:] = slope[num_diag:] - slope[:-num_diag]
        i_diag = np.nonzero(slope == num_diag)[0]
        if len(i_diag) == 0:
            return find_cue(wp, cue_in, num_diag // 2)
        else:
            i = i_diag[-1]
        return x[i] + 1, y[i] + 1


def extend_wp(wp, total_wpts):
    """Extend a warp path with diagonal padding at both ends.

    Pads before the first matched point and after the last matched point
    so that the extended path covers the full track length.

    Args:
        wp: warp path array (N, 2), descending order.
        total_wpts: total number of warp points in the track.

    Returns:
        Extended warp path array.
    """
    num_pad_befor = wp[-1, 0]
    num_pad_after = total_wpts - wp[0, 0] - 1

    if num_pad_befor > 0:
        pad_befor = np.stack([
            np.arange(0, num_pad_befor),
            np.arange(wp[-1, 1] - num_pad_befor, wp[-1, 1]),
        ], axis=1)[::-1]
    else:
        pad_befor = np.empty((0, 2), dtype=wp.dtype)

    if num_pad_after > 0:
        pad_after = np.stack([
            np.arange(wp[0, 0] + 1, wp[0, 0] + 1 + num_pad_after),
            np.arange(wp[0, 1] + 1, wp[0, 1] + 1 + num_pad_after),
        ], axis=1)[::-1]
    else:
        pad_after = np.empty((0, 2), dtype=wp.dtype)

    wp_ext = np.concatenate([pad_after, wp, pad_befor])
    return wp_ext


def anchors(wp, start_wpt_mix, last_wpt_mix, halfbeats_mix, halfbeats_trk):
    """Compute anchor points for time-scale modification.

    Maps mix warp-point indices to track warp-point indices through the
    warp path, then looks up the corresponding halfbeat times.

    Args:
        wp: extended warp path array (N, 2).
        start_wpt_mix: starting mix warp-point index.
        last_wpt_mix: ending mix warp-point index.
        halfbeats_mix: halfbeat times for the mix.
        halfbeats_trk: halfbeat times for the track.

    Returns:
        (anchors_trk, anchors_mix) -- arrays of anchor times in seconds.
    """
    mix2trk_wpt = {wpt_mix: wpt_trk for wpt_trk, wpt_mix in wp}

    start_wpt = mix2trk_wpt[start_wpt_mix]
    last_wpt = mix2trk_wpt[last_wpt_mix]

    wpt_trk, wpt_mix = wp[::-1].T
    i_start_wpt, i_last_wpt = np.searchsorted(wpt_trk, [start_wpt, last_wpt])

    i_anchors_trk = wpt_trk[i_start_wpt:i_last_wpt + 1]
    i_anchors_mix = wpt_mix[i_start_wpt:i_last_wpt + 1]

    anchors_trk = halfbeats_trk[i_anchors_trk]
    anchors_mix = halfbeats_mix[i_anchors_mix]

    # Include the end beat (with bounds check).
    end_idx_trk = min(i_anchors_trk[-1] + 1, len(halfbeats_trk) - 1)
    end_idx_mix = min(i_anchors_mix[-1] + 1, len(halfbeats_mix) - 1)
    anchors_trk = np.concatenate([anchors_trk, [halfbeats_trk[end_idx_trk]]])
    anchors_mix = np.concatenate([anchors_mix, [halfbeats_mix[end_idx_mix]]])

    return anchors_trk, anchors_mix


def project_wp_raw(wp, track_cue_in, track_cue_out, mix_cue_in, mix_cue_out):
    """Project warp points with diagonal padding outside the cue region.

    Args:
        wp: warp path array (N, 2), descending order.
        track_cue_in: track cue-in warp-point index.
        track_cue_out: track cue-out warp-point index.
        mix_cue_in: mix cue-in warp-point index.
        mix_cue_out: mix cue-out warp-point index.

    Returns:
        Extended warp path array with padding.
    """
    pad_stop_befor = track_cue_in
    if pad_stop_befor > 0:
        pad_befor = np.stack([
            np.arange(0, pad_stop_befor),
            np.arange(mix_cue_in - pad_stop_befor, mix_cue_in),
        ], axis=1)[::-1]
    else:
        pad_befor = np.empty((0, 2), dtype=wp.dtype)

    track_total_beats = wp[0, 0] + 1
    pad_stop_after = track_total_beats - track_cue_out
    if pad_stop_after > 0:
        pad_after = np.stack([
            np.arange(track_cue_out + 1, track_cue_out + pad_stop_after),
            np.arange(mix_cue_out + 1, mix_cue_out + pad_stop_after),
        ], axis=1)[::-1]
    else:
        pad_after = np.empty((0, 2), dtype=wp.dtype)

    wp_ext = wp[(track_cue_in <= wp[:, 0]) & (wp[:, 0] <= track_cue_out)]
    wp_ext = np.concatenate([pad_after, wp_ext, pad_befor])

    return wp_ext
