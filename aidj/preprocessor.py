from __future__ import annotations

import logging

import numpy as np
import librosa
import soundfile as sf

from aidj import config
from aidj.analyzer import TrackInfo

log = logging.getLogger(__name__)


def detect_cue_points(track_info: TrackInfo) -> tuple[float, float]:
    """Detect cue_out (for outgoing track) and cue_in (for incoming track).

    cue_out: where the track starts winding down (outro)
    cue_in: where the track finishes its intro and gets going

    Returns (cue_out_seconds, cue_in_seconds).
    """
    duration = track_info.duration
    boundaries = track_info.structure_boundaries

    # Try to find cue_out: last structural boundary in the final 30% of the track
    cue_out = None
    for b in reversed(boundaries):
        if 0.65 * duration < b < 0.95 * duration:
            cue_out = b
            break
    if cue_out is None:
        cue_out = duration * config.CUE_OUT_PERCENT

    # Try to find cue_in: first structural boundary in the first 30% of the track
    cue_in = None
    for b in boundaries:
        if 0.05 * duration < b < 0.35 * duration:
            cue_in = b
            break
    if cue_in is None:
        cue_in = duration * config.CUE_IN_PERCENT

    return cue_out, cue_in


def _snap_to_beat(time_sec: float, beat_positions: list[float]) -> float:
    if not beat_positions:
        return time_sec
    beats = np.array(beat_positions)
    idx = np.argmin(np.abs(beats - time_sec))
    return float(beats[idx])


def sync_bpm(
    audio_b: np.ndarray,
    bpm_a: float,
    bpm_b: float,
    sr: int = config.SR,
) -> np.ndarray:
    """Time-stretch audio_b to match bpm_a. Returns stretched audio."""
    if abs(bpm_a - bpm_b) < 0.5:
        return audio_b
    ratio = bpm_a / bpm_b
    log.info(f"Time-stretching BPM {bpm_b:.1f} → {bpm_a:.1f} (ratio={ratio:.3f})")
    try:
        import pyrubberband as pyrb
        stretched = pyrb.time_stretch(audio_b, sr, ratio)
    except ImportError:
        log.warning("pyrubberband not available, using librosa time_stretch")
        stretched = librosa.effects.time_stretch(audio_b, rate=1.0 / ratio)
    return stretched


def prepare_pair(
    track_a: TrackInfo,
    track_b: TrackInfo,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load and prepare a track pair for transition generation.

    Returns (audio_a, audio_b, cue_out_a, cue_in_b) where:
    - audio_a, audio_b are mono float32 arrays at config.SR
    - cue_out_a is the cue-out point in track A (seconds)
    - cue_in_b is the cue-in point in track B (seconds)
    """
    audio_a, _ = librosa.load(track_a.path, sr=config.SR, mono=True)
    audio_b, _ = librosa.load(track_b.path, sr=config.SR, mono=True)

    cue_out_a, _ = detect_cue_points(track_a)
    _, cue_in_b = detect_cue_points(track_b)

    # Snap cues to nearest beats
    cue_out_a = _snap_to_beat(cue_out_a, track_a.beat_positions)
    cue_in_b = _snap_to_beat(cue_in_b, track_b.beat_positions)

    # Sync BPM: stretch track B to match track A
    audio_b = sync_bpm(audio_b, track_a.bpm, track_b.bpm)

    return audio_a, audio_b, cue_out_a, cue_in_b
