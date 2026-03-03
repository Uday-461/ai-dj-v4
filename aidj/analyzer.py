from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field, asdict

import numpy as np
import librosa
import soundfile as sf

from aidj import config
from aidj.camelot import pitch_class_to_key, key_to_camelot

log = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    path: str
    bpm: float
    key: str  # e.g. "C major", "A minor"
    camelot_code: str | None  # e.g. "8B", "8A"
    energy: float  # 0-1 normalized RMS energy
    duration: float  # seconds
    beat_positions: list[float] = field(default_factory=list)
    structure_boundaries: list[float] = field(default_factory=list)


def _estimate_key(y: np.ndarray, sr: int) -> str:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Major and minor profiles (Krumhansl-Kessler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_corr = -1
    best_pitch = 0
    best_is_minor = False

    for shift in range(12):
        rotated = np.roll(chroma_mean, -shift)
        corr_major = np.corrcoef(rotated, major_profile)[0, 1]
        corr_minor = np.corrcoef(rotated, minor_profile)[0, 1]
        if corr_major > best_corr:
            best_corr = corr_major
            best_pitch = shift
            best_is_minor = False
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_pitch = shift
            best_is_minor = True

    return pitch_class_to_key(best_pitch, best_is_minor)


def _estimate_energy(y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    # Normalize: typical RMS for music is 0.01-0.3
    normalized = np.clip(mean_rms / 0.2, 0.0, 1.0)
    return float(normalized)


def _detect_structure_boundaries(y: np.ndarray, sr: int) -> list[float]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Spectral novelty for structure detection
    novelty = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    # Find peaks in novelty that are strong enough to be structural
    peaks = librosa.util.peak_pick(novelty, pre_max=30, post_max=30,
                                   pre_avg=100, post_avg=100,
                                   delta=0.1, wait=100)
    times = librosa.frames_to_time(peaks, sr=sr)
    return times.tolist()


def analyze_track(path: str) -> TrackInfo:
    log.info(f"Analyzing {os.path.basename(path)}")
    y, sr = librosa.load(path, sr=config.SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # BPM and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    beat_positions = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Key
    key = _estimate_key(y, sr)
    camelot_code = key_to_camelot(key)

    # Energy
    energy = _estimate_energy(y)

    # Structure boundaries
    structure_boundaries = _detect_structure_boundaries(y, sr)

    return TrackInfo(
        path=os.path.abspath(path),
        bpm=bpm,
        key=key,
        camelot_code=camelot_code,
        energy=energy,
        duration=duration,
        beat_positions=beat_positions,
        structure_boundaries=structure_boundaries,
    )


def analyze_library(folder: str, cache_path: str | None = None) -> list[TrackInfo]:
    # Load cache if it exists
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            for item in json.load(f):
                cache[item["path"]] = item

    tracks = []
    for root, _, files in os.walk(folder):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in config.AUDIO_EXTENSIONS:
                continue
            fpath = os.path.abspath(os.path.join(root, fname))
            if fpath in cache:
                tracks.append(TrackInfo(**cache[fpath]))
                log.info(f"Cached: {fname}")
            else:
                try:
                    info = analyze_track(fpath)
                    tracks.append(info)
                except Exception as e:
                    log.warning(f"Failed to analyze {fname}: {e}")

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump([asdict(t) for t in tracks], f, indent=2)

    return tracks
