from __future__ import annotations

import logging

from aidj.analyzer import TrackInfo
from aidj.camelot import is_compatible
from aidj import config

log = logging.getLogger(__name__)


def _bpm_similarity(a: TrackInfo, b: TrackInfo) -> float:
    diff = abs(a.bpm - b.bpm)
    if diff > config.MAX_BPM_DIFF:
        return 0.0
    return 1.0 - diff / config.MAX_BPM_DIFF


def _key_match(a: TrackInfo, b: TrackInfo) -> float:
    if a.camelot_code == b.camelot_code:
        return 1.0
    if is_compatible(a.camelot_code, b.camelot_code):
        return 0.7
    return 0.0


def _energy_flow(a: TrackInfo, b: TrackInfo) -> float:
    diff = abs(a.energy - b.energy)
    if diff > config.MAX_ENERGY_DIFF:
        return 0.0
    # Slight bonus for energy increase (building energy)
    if b.energy >= a.energy:
        return 1.0 - diff / config.MAX_ENERGY_DIFF
    return 0.8 * (1.0 - diff / config.MAX_ENERGY_DIFF)


def _duration_compatibility(a: TrackInfo, b: TrackInfo) -> float:
    # Prefer tracks of similar length (both long enough for transitions)
    min_dur = min(a.duration, b.duration)
    if min_dur < 90:  # too short for a good transition
        return 0.3
    ratio = min(a.duration, b.duration) / max(a.duration, b.duration)
    return ratio


def score_pair(a: TrackInfo, b: TrackInfo) -> float:
    w = config.SCORE_WEIGHTS
    bpm = _bpm_similarity(a, b)
    key = _key_match(a, b)
    energy = _energy_flow(a, b)
    dur = _duration_compatibility(a, b)

    # Hard filters
    if bpm == 0.0:
        return 0.0
    if key == 0.0:
        return 0.0
    if abs(a.energy - b.energy) > config.MAX_ENERGY_DIFF:
        return 0.0

    return bpm * w["bpm"] + key * w["key"] + energy * w["energy"] + dur * w["duration"]


def build_playlist(
    tracks: list[TrackInfo],
    start_track: TrackInfo | None = None,
) -> list[TrackInfo]:
    if len(tracks) < 2:
        return list(tracks)

    remaining = set(range(len(tracks)))

    if start_track is not None:
        # Find the start track by path
        current_idx = next(
            (i for i, t in enumerate(tracks) if t.path == start_track.path),
            0,
        )
    else:
        # Start with a track near median energy (good warmup start)
        energies = [t.energy for t in tracks]
        median_e = sorted(energies)[len(energies) // 4]  # lower quartile
        current_idx = min(range(len(tracks)), key=lambda i: abs(tracks[i].energy - median_e))

    playlist = [tracks[current_idx]]
    remaining.remove(current_idx)

    while remaining:
        current = playlist[-1]
        best_score = -1
        best_idx = None
        for idx in remaining:
            s = score_pair(current, tracks[idx])
            if s > best_score:
                best_score = s
                best_idx = idx

        if best_idx is None or best_score == 0.0:
            # No compatible track found, pick closest BPM
            best_idx = min(remaining, key=lambda i: abs(tracks[i].bpm - current.bpm))
            log.warning(
                f"No compatible match for {current.path}, "
                f"falling back to closest BPM: {tracks[best_idx].path}"
            )

        playlist.append(tracks[best_idx])
        remaining.remove(best_idx)

    return playlist
