from __future__ import annotations

import logging
import os

import numpy as np
import librosa

from aidj import config

log = logging.getLogger(__name__)


class BeatDetector:
    """Beat and downbeat detection with statistical correction.

    Replaces the madmom-based pipeline from djmix-dataset with
    BeatNet (preferred) or librosa (fallback).
    """

    def __init__(self, method="beatnet"):
        self.method = method
        self._beatnet = None
        if method == "beatnet":
            try:
                from BeatNet.BeatNet import BeatNet
                self._beatnet = BeatNet(
                    1,  # model number (1 = best for offline)
                    mode='offline',
                    inference_model='DBN',
                    plot=[],
                    thread=False,
                )
            except ImportError:
                log.warning("BeatNet not available, falling back to librosa")
                self.method = "librosa"

    def detect(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Detect beats and downbeats.

        Returns:
            (beat_times, downbeat_times) as 1D numpy arrays in seconds.
        """
        if self.method == "beatnet" and self._beatnet is not None:
            return self._detect_beatnet(audio_path)
        return self._detect_librosa(audio_path)

    def _detect_beatnet(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Use BeatNet for beat/downbeat detection.

        BeatNet returns array of shape (N, 2) where:
        - column 0: time in seconds
        - column 1: beat position within bar (1 = downbeat)
        """
        result = self._beatnet.process(audio_path)
        if result is None or len(result) == 0:
            log.warning(f"BeatNet returned no beats for {audio_path}")
            return np.array([]), np.array([])

        beat_times = result[:, 0]
        downbeat_mask = result[:, 1] == 1
        downbeat_times = beat_times[downbeat_mask]

        return beat_times, downbeat_times

    def _detect_librosa(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: use librosa for beat detection (no downbeat info)."""
        y, sr = librosa.load(audio_path, sr=config.FEATURE_SR, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Librosa doesn't detect downbeats; estimate every 4th beat
        downbeat_times = beat_times[::4] if len(beat_times) >= 4 else beat_times

        return beat_times, downbeat_times

    def corrected_beats(self, audio_path: str, num_beats_aside: int = 33) -> tuple[np.ndarray, np.ndarray]:
        """Detect beats and apply statistical outlier correction.

        Port of djmix/features/beats.py:corrected_beats().

        Finds beats with anomalous intervals (missing or extra beats)
        and replaces them with evenly spaced beats based on surrounding
        median intervals.

        Returns:
            (corrected_beat_times, downbeat_times) tuple.
        """
        beat_times, downbeat_times = self.detect(audio_path)

        if len(beat_times) < 4:
            return beat_times, downbeat_times

        beats = beat_times
        intvls = np.diff(beats).round(2)

        # Sliding window median absolute deviation to find outliers
        padded = librosa.util.pad_center(
            intvls,
            size=len(intvls) + num_beats_aside - 1,
            mode='reflect',
        )
        frames = librosa.util.frame(
            padded,
            frame_length=num_beats_aside,
            hop_length=1,
        )

        absdev = np.abs(frames - np.median(frames, axis=0))
        scores = absdev[num_beats_aside // 2] / 0.01
        scores = np.nan_to_num(scores, nan=0, posinf=0, neginf=0)
        outliers = np.flatnonzero(scores > 2)

        if outliers.size == 0:
            return beats, downbeat_times

        # Group consecutive outlier indices
        outlier_groups = np.split(
            outliers, np.flatnonzero(np.diff(outliers) != 1) + 1
        )

        segments = []
        i_working = 0

        for inds in outlier_groups:
            first = inds[0] - 1
            last = inds[-1] + 1

            if (first - 1) < 0 or (last + 2) > len(beats):
                continue

            l_intvl = np.median(
                intvls[max(first - num_beats_aside // 2, 0):first]
            )
            r_intvl = np.median(
                intvls[last:min(last + num_beats_aside // 2, len(intvls))]
            )
            c_intvl = (l_intvl + r_intvl) / 2

            segment_intvls = intvls[first:last + 1]
            num_beats_crt = round(segment_intvls.sum() / c_intvl)
            segment_beats_crt = np.linspace(
                beats[first], beats[last], num_beats_crt
            )
            segment_beats_crt = segment_beats_crt[:-1]

            segments.append(beats[i_working:first])
            segments.append(segment_beats_crt)
            i_working = last

        segments.append(beats[i_working:])
        beats_crt = np.concatenate(segments)

        return beats_crt, downbeat_times


def beat_aggregate(feature, beats, sr, hop_length, frames_per_beat=None):
    """Aggregate a feature matrix at beat resolution.

    Port of djmix/features/beats.py:beat_aggregate().

    Args:
        feature: (n_features, n_frames) feature matrix
        beats: beat times in seconds
        sr: sample rate
        hop_length: hop length used to compute features
        frames_per_beat: if set, resize each beat segment to this many frames.
            If None, average features per beat.

    Returns:
        Beat-aggregated feature matrix.
    """
    from skimage.transform import resize as sk_resize

    max_frame = feature.shape[1]
    beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop_length)
    beat_frames = beat_frames[beat_frames < max_frame]
    beat_feature = np.split(feature, beat_frames, axis=1)
    # Only use features between beats (not before first or after last)
    beat_feature = beat_feature[1:-1]

    if frames_per_beat is not None:
        beat_feature = [
            sk_resize(f, (f.shape[0], frames_per_beat))
            for f in beat_feature
        ]
        beat_feature = np.concatenate(beat_feature, axis=1)
    else:
        beat_feature = [f.mean(axis=1) for f in beat_feature]
        beat_feature = np.array(beat_feature).T

    return beat_feature


def compute_halfbeat_chroma(audio_path: str, beats: np.ndarray, sr: int = None, hop: int = None, y: np.ndarray = None):
    """Compute chroma features at half-beat resolution.

    Returns (12, n_halfbeats) chroma matrix.
    Pass pre-loaded audio array as `y` to avoid redundant disk reads.
    Defaults to FEATURE_SR (22050) and a proportional hop for efficiency.
    """
    if sr is None:
        sr = config.FEATURE_SR
    if hop is None:
        hop = 512  # ~23ms at 22050Hz, same temporal resolution as 1024@44100

    if y is None:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # CQT-based chroma — fmin=C1 (32.7Hz), 7 octaves (C1-C8, max ~4186Hz),
    # well within Nyquist at FEATURE_SR=22050Hz (11025Hz ceiling).
    fmin = librosa.note_to_hz('C1')  # 32.7 Hz
    n_bins = 84  # 7 octaves * 12 bins/octave
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=fmin,
                              n_bins=n_bins, bins_per_octave=12))
    chroma = librosa.feature.chroma_cens(
        C=cqt, hop_length=hop, fmin=fmin,
        n_chroma=12, n_octaves=7, bins_per_octave=12,
    )

    halfbeat_chroma = beat_aggregate(chroma, beats, sr, hop, frames_per_beat=2)
    return halfbeat_chroma


def compute_halfbeat_mfcc(audio_path: str, beats: np.ndarray, sr: int = None, hop: int = None, n_mfcc: int = 20, y: np.ndarray = None):
    """Compute MFCC features at half-beat resolution.

    Returns (n_mfcc, n_halfbeats) MFCC matrix.
    Pass pre-loaded audio array as `y` to avoid redundant disk reads.
    Defaults to FEATURE_SR (22050) and a proportional hop for efficiency.
    """
    if sr is None:
        sr = config.FEATURE_SR
    if hop is None:
        hop = 512  # ~23ms at 22050Hz, same temporal resolution as 1024@44100

    if y is None:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=hop,
        n_mels=128, fmin=20, fmax=sr // 2, power=1,
    )
    mfcc = librosa.feature.mfcc(S=librosa.amplitude_to_db(mel_spec), n_mfcc=n_mfcc)

    halfbeat_mfcc = beat_aggregate(mfcc, beats, sr, hop, frames_per_beat=2)
    return halfbeat_mfcc
