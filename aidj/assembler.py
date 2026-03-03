from __future__ import annotations

import logging
import os

import numpy as np
import soundfile as sf

from aidj import config

log = logging.getLogger(__name__)


def _crossfade(audio_a: np.ndarray, audio_b: np.ndarray, fade_samples: int = 441) -> np.ndarray:
    """Apply a short crossfade between two audio segments to avoid clicks.

    Default is 441 samples = 10ms at 44100 Hz.

    Uses linear fade curves. The last `fade_samples` of audio_a are blended
    with the first `fade_samples` of audio_b, reducing the total output length
    by `fade_samples` relative to a simple concatenation.
    """
    if len(audio_a) == 0:
        return audio_b.copy()
    if len(audio_b) == 0:
        return audio_a.copy()

    # Clamp fade to the shorter of the two segments
    fade_samples = min(fade_samples, len(audio_a), len(audio_b))

    if fade_samples <= 0:
        return np.concatenate([audio_a, audio_b])

    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

    # Region before the overlap
    head = audio_a[: len(audio_a) - fade_samples]
    # Overlapping region: blend tail of a with head of b
    overlap = audio_a[len(audio_a) - fade_samples :] * fade_out + audio_b[:fade_samples] * fade_in
    # Region after the overlap
    tail = audio_b[fade_samples:]

    return np.concatenate([head, overlap, tail])


def _lufs_normalize(
    audio: np.ndarray,
    sr: int = config.SR,
    target_lufs: float = config.TARGET_LUFS,
) -> np.ndarray:
    """Normalize audio to target LUFS level.

    Uses pyloudnorm for ITU-R BS.1770 integrated loudness measurement.
    Falls back to peak normalization if the audio is too short or silent.
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        log.warning("pyloudnorm not installed; skipping LUFS normalization")
        return audio

    if audio.size == 0:
        return audio

    # pyloudnorm expects (samples,) or (samples, channels); ensure float64
    data = audio.astype(np.float64)
    if data.ndim == 1:
        # Meter expects 2-D: (samples, channels)
        data_2d = data[:, np.newaxis]
    else:
        data_2d = data

    meter = pyln.Meter(sr)  # ITU-R BS.1770-4

    try:
        loudness = meter.integrated_loudness(data_2d)
    except Exception as exc:
        log.warning("LUFS measurement failed (%s); skipping normalization", exc)
        return audio

    if not np.isfinite(loudness):
        log.warning("Audio is silent or too short to measure LUFS; skipping normalization")
        return audio

    gain_db = target_lufs - loudness
    gain_linear = 10.0 ** (gain_db / 20.0)

    log.info(
        "LUFS normalization: measured=%.1f LUFS, target=%.1f LUFS, gain=%.2f dB",
        loudness,
        target_lufs,
        gain_db,
    )

    normalized = audio * gain_linear

    # Hard clip guard to prevent digital overs
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        log.warning("Peak after normalization is %.4f; applying peak limiting", peak)
        normalized = normalized / peak * 0.9999

    return normalized.astype(np.float32)


def assemble_mix(
    playlist_audio: list[np.ndarray],
    transitions: list[np.ndarray],
    cue_outs: list[float],
    cue_ins: list[float],
    sr: int = config.SR,
) -> np.ndarray:
    """Assemble a continuous mix from individual tracks and transitions.

    Args:
        playlist_audio: list of mono audio arrays for each track in order.
        transitions: list of transition audio arrays (len = len(playlist) - 1).
            Each transition is the mixed window produced by DJtransGAN.
        cue_outs: list of cue-out times in seconds for each track.
        cue_ins: list of cue-in times in seconds for each track.
        sr: sample rate.

    Returns:
        The assembled mix as a mono float32 numpy array, LUFS-normalized.
    """
    n_tracks = len(playlist_audio)

    if n_tracks == 0:
        log.warning("Empty playlist; returning empty array")
        return np.zeros(0, dtype=np.float32)

    if n_tracks == 1:
        log.info("Single-track playlist; returning full track after normalization")
        track = playlist_audio[0].astype(np.float32)
        return _lufs_normalize(track, sr)

    expected_transitions = n_tracks - 1
    if len(transitions) != expected_transitions:
        log.warning(
            "Expected %d transitions for %d tracks, got %d; padding with silence",
            expected_transitions,
            n_tracks,
            len(transitions),
        )
        # Pad missing transitions with silence sized to a typical transition window
        silence_len = config.N_TIME * sr
        while len(transitions) < expected_transitions:
            transitions = list(transitions) + [np.zeros(silence_len, dtype=np.float32)]

    # Short crossfade length: 10 ms
    fade_samples = int(0.010 * sr)

    segments: list[np.ndarray] = []

    for i in range(n_tracks):
        track = playlist_audio[i].astype(np.float32)
        track_len = len(track)

        cue_out_sample = int(cue_outs[i] * sr)
        cue_in_sample = int(cue_ins[i] * sr)

        # Clamp cue points to valid range
        cue_out_sample = max(0, min(cue_out_sample, track_len))
        cue_in_sample = max(0, min(cue_in_sample, track_len))

        is_first = i == 0
        is_last = i == n_tracks - 1

        # ------------------------------------------------------------------
        # Solo portion of this track
        # For the first track: from the very beginning to cue_out.
        # For middle/last tracks: from cue_in to cue_out (or end if last).
        # ------------------------------------------------------------------
        if is_first:
            solo_start = 0
        else:
            solo_start = cue_in_sample

        if is_last:
            solo_end = track_len
        else:
            solo_end = cue_out_sample

        solo_end = max(solo_start, solo_end)
        solo = track[solo_start:solo_end]

        log.info(
            "Track %d/%d: solo portion [%.2fs – %.2fs] = %d samples",
            i + 1,
            n_tracks,
            solo_start / sr,
            solo_end / sr,
            len(solo),
        )

        if len(segments) == 0:
            segments.append(solo)
        else:
            # Crossfade the previous content into this solo segment
            assembled_so_far = segments[-1]
            merged = _crossfade(assembled_so_far, solo, fade_samples)
            segments[-1] = merged

        # ------------------------------------------------------------------
        # Transition to the next track (if not last)
        # ------------------------------------------------------------------
        if not is_last:
            transition = transitions[i].astype(np.float32)

            log.info(
                "Transition %d->%d: %d samples (%.2fs)",
                i + 1,
                i + 2,
                len(transition),
                len(transition) / sr,
            )

            # Crossfade current mix tail into transition
            current = segments[-1]
            merged = _crossfade(current, transition, fade_samples)
            segments[-1] = merged

    # Flatten segments list (should already be a single accumulated array)
    mix = segments[-1] if segments else np.zeros(0, dtype=np.float32)

    log.info(
        "Raw mix assembled: %d samples (%.2f seconds)",
        len(mix),
        len(mix) / sr,
    )

    mix = _lufs_normalize(mix, sr)

    log.info("Mix assembly complete")
    return mix


def save_mix(audio: np.ndarray, path: str, sr: int = config.SR):
    """Save the mix to a WAV or MP3 file.

    soundfile does not support MP3 writing. If an MP3 path is requested
    this function will attempt to use pydub/ffmpeg, and if unavailable
    will fall back to saving a WAV file with a warning.

    Args:
        audio: mono float32 numpy array.
        path: destination file path (.wav or .mp3).
        sr: sample rate.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".mp3":
        saved = _save_mp3(audio, path, sr)
        if not saved:
            wav_path = os.path.splitext(path)[0] + ".wav"
            log.warning(
                "MP3 export not available; saving as WAV instead: %s", wav_path
            )
            path = wav_path
            ext = ".wav"

    if ext != ".mp3":
        _save_wav(audio, path, sr)


def _save_wav(audio: np.ndarray, path: str, sr: int):
    """Write audio to a WAV file using soundfile."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    log.info("Saving WAV: %s (%d samples, %d Hz)", path, len(audio), sr)
    sf.write(path, audio.astype(np.float32), sr, subtype="FLOAT")
    log.info("WAV saved: %s", path)


def _save_mp3(audio: np.ndarray, path: str, sr: int) -> bool:
    """Attempt to write audio to an MP3 file via pydub + ffmpeg.

    Returns True on success, False if pydub or ffmpeg is unavailable.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        log.warning("pydub not installed; cannot write MP3")
        return False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # pydub works with 16-bit PCM integers
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        segment = AudioSegment(
            pcm.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )
        segment.export(path, format="mp3", bitrate="320k")
        log.info("MP3 saved: %s", path)
        return True
    except Exception as exc:
        log.warning("MP3 export failed (%s)", exc)
        return False
