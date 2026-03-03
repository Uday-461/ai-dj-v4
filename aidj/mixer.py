from __future__ import annotations

import logging

import numpy as np
import librosa
from scipy.signal import sosfilt

from aidj import config
from aidj.curves.eq_filters import eq3_filters

log = logging.getLogger(__name__)


def apply_stem_curves(
    stems: dict[str, np.ndarray],
    curves: dict[str, np.ndarray],
    sr: int = config.SR,
    hop_size: int = config.OPT_HOP,
) -> np.ndarray:
    """Apply fader + EQ curves to stems and sum.

    Args:
        stems: dict mapping stem name -> 1D audio array
        curves: dict mapping "{stem}_{param}" -> 1D curve array
            Params: fader_prev/next, eq_prev/next_low/mid/high
        sr: sample rate
        hop_size: hop size used for curve computation

    Returns:
        Mixed audio as 1D numpy array
    """
    # Determine output length from the longest stem
    max_len = max(len(v) for v in stems.values())
    output = np.zeros(max_len, dtype=np.float32)

    for stem_name, audio in stems.items():
        audio = audio.astype(np.float32)

        # Get fader curve for this stem
        fader_key = f"{stem_name}_fader"
        fader = None
        for suffix in ['_prev', '_next']:
            key = f"{stem_name}_fader{suffix}"
            if key in curves:
                if fader is None:
                    fader = curves[key]
                else:
                    # For the mixed output, we might have both prev and next
                    # In practice, apply_deck_curves is called per-deck
                    pass

        if fader is not None:
            audio = _apply_fader(audio, fader, hop_size)

        # Get EQ curves
        eq_low_key = f"{stem_name}_eq_low"
        eq_mid_key = f"{stem_name}_eq_mid"
        eq_high_key = f"{stem_name}_eq_high"

        # Check for any EQ key pattern
        for suffix in ['_prev', '_next']:
            low_key = f"{stem_name}_eq{suffix}_low"
            mid_key = f"{stem_name}_eq{suffix}_mid"
            high_key = f"{stem_name}_eq{suffix}_high"

            if low_key in curves and mid_key in curves and high_key in curves:
                audio = _apply_eq(
                    audio, curves[low_key], curves[mid_key], curves[high_key],
                    sr=sr, hop_size=hop_size,
                )
                break

        # Add to output (pad if needed)
        out_len = min(len(audio), max_len)
        output[:out_len] += audio[:out_len]

    return output


def apply_deck_curves(
    stems: dict[str, np.ndarray],
    curves: dict[str, np.ndarray],
    deck: str,
    sr: int = config.SR,
    hop_size: int = config.OPT_HOP,
) -> np.ndarray:
    """Apply curves for a specific deck (prev or next) to all stems.

    Args:
        stems: dict mapping stem name -> 1D audio array
        curves: dict mapping "{stem}_{param}" -> 1D curve array
        deck: "prev" or "next"
        sr: sample rate
        hop_size: hop size

    Returns:
        Mixed audio for this deck as 1D numpy array
    """
    max_len = max(len(v) for v in stems.values())
    output = np.zeros(max_len, dtype=np.float32)

    for stem_name, audio in stems.items():
        audio = audio.astype(np.float32)

        # Apply fader
        fader_key = f"{stem_name}_fader_{deck}"
        if fader_key in curves:
            audio = _apply_fader(audio, curves[fader_key], hop_size)

        # Apply EQ
        eq_low_key = f"{stem_name}_eq_{deck}_low"
        eq_mid_key = f"{stem_name}_eq_{deck}_mid"
        eq_high_key = f"{stem_name}_eq_{deck}_high"

        if all(k in curves for k in [eq_low_key, eq_mid_key, eq_high_key]):
            audio = _apply_eq(
                audio,
                curves[eq_low_key], curves[eq_mid_key], curves[eq_high_key],
                sr=sr, hop_size=hop_size,
            )

        out_len = min(len(audio), max_len)
        output[:out_len] += audio[:out_len]

    return output


def mix_transition(
    prev_stems: dict[str, np.ndarray],
    next_stems: dict[str, np.ndarray],
    curves: dict[str, np.ndarray],
    sr: int = config.SR,
    hop_size: int = config.OPT_HOP,
) -> np.ndarray:
    """Mix a transition from stem audio and predicted curves.

    Args:
        prev_stems: stems from the outgoing track
        next_stems: stems from the incoming track
        curves: predicted curves with keys like "{stem}_fader_prev", etc.
        sr: sample rate
        hop_size: hop size

    Returns:
        Mixed transition audio as 1D numpy array
    """
    prev_mixed = apply_deck_curves(prev_stems, curves, "prev", sr, hop_size)
    next_mixed = apply_deck_curves(next_stems, curves, "next", sr, hop_size)

    # Sum the two decks
    max_len = max(len(prev_mixed), len(next_mixed))
    output = np.zeros(max_len, dtype=np.float32)
    output[:len(prev_mixed)] += prev_mixed
    output[:len(next_mixed)] += next_mixed

    # Clip to prevent digital overs
    peak = np.max(np.abs(output))
    if peak > 1.0:
        log.warning(f"Transition peak {peak:.3f} > 1.0, applying limiter")
        output = output / peak * 0.9999

    return output


def _apply_fader(audio, fader_curve, hop_size):
    """Apply a gain curve to audio using overlap-add.

    Args:
        audio: 1D audio array
        fader_curve: 1D gain values per frame
        hop_size: samples per frame

    Returns:
        Gain-adjusted audio
    """
    n_samples = len(audio)
    n_frames = len(fader_curve)
    window_size = 2 * hop_size

    audio = audio.astype(np.float32)
    audio_padded = librosa.util.fix_length(audio, size=n_samples + window_size)

    window = np.hanning(window_size).astype(np.float32)
    output = np.zeros_like(audio_padded)

    for i in range(min(n_frames, (len(audio_padded) - window_size) // hop_size + 1)):
        start = i * hop_size
        end = start + window_size
        if end > len(audio_padded):
            break
        segment = audio_padded[start:end] * fader_curve[i]
        output[start:end] += window * segment

    # Remove padding
    lpad = window_size // 2
    output = output[lpad:lpad + n_samples]

    return output


def _apply_eq(audio, eq_low_db, eq_mid_db, eq_high_db, sr, hop_size):
    """Apply time-varying 3-band EQ to audio.

    Args:
        audio: 1D audio array
        eq_low_db: per-frame low band gain in dB
        eq_mid_db: per-frame mid band gain in dB
        eq_high_db: per-frame high band gain in dB
        sr: sample rate
        hop_size: samples per frame

    Returns:
        EQ-processed audio
    """
    n_samples = len(audio)
    n_frames = min(len(eq_low_db), len(eq_mid_db), len(eq_high_db))
    window_size = 2 * hop_size

    # Convert EQ curves from linear ratio to dB for filter design
    # The curves from the optimizer are linear gain ratios (0 to 1)
    # Convert to dB: 0.0 -> -80dB, 1.0 -> 0dB
    def ratio_to_db(ratio, min_db=-80):
        ratio = np.clip(ratio, 1e-4, 2.0)
        return 20 * np.log10(ratio)

    audio = audio.astype(np.float32)
    audio_padded = librosa.util.fix_length(audio, size=n_samples + window_size)

    window = np.hanning(window_size).astype(np.float32)
    output = np.zeros_like(audio_padded)

    for i in range(min(n_frames, (len(audio_padded) - window_size) // hop_size + 1)):
        start = i * hop_size
        end = start + window_size
        if end > len(audio_padded):
            break

        segment = audio_padded[start:end].copy()

        # Build per-frame EQ filters
        low_db = float(ratio_to_db(eq_low_db[i]))
        mid_db = float(ratio_to_db(eq_mid_db[i]))
        high_db = float(ratio_to_db(eq_high_db[i]))

        # Only apply EQ if gains are significantly different from unity
        if abs(low_db) > 0.5 or abs(mid_db) > 0.5 or abs(high_db) > 0.5:
            sos_low, sos_mid, sos_high = eq3_filters(
                low_db_gain=low_db,
                mid_db_gain=mid_db,
                high_db_gain=high_db,
                cutoff_low=config.EQ_CUTOFF_LOW,
                center_mid=config.EQ_CENTER_MID,
                cutoff_high=config.EQ_CUTOFF_HIGH,
                sr=sr,
            )
            segment = sosfilt(sos_low, segment)
            segment = sosfilt(sos_mid, segment)
            segment = sosfilt(sos_high, segment)

        output[start:end] += window * segment

    lpad = window_size // 2
    output = output[lpad:lpad + n_samples]

    return output
