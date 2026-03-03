from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import librosa

from aidj import config
from aidj.curves.optimizer import EQ3FaderOptimizer, OptConfig, compute_spectrogram

log = logging.getLogger(__name__)


class StemCurveExtractor:
    """Extract per-stem fader + EQ curves from transitions.

    For each transition, loads the time-aligned stem audio for the mix,
    previous track, and next track, then runs convex optimization
    per-stem to extract curves.
    """

    def __init__(self, opt_config: OptConfig = None):
        self.opt_config = opt_config or OptConfig()
        self.optimizer = EQ3FaderOptimizer(self.opt_config)

    def extract_transition_curves(
        self,
        mix_stems: dict[str, np.ndarray],
        prev_stems: dict[str, np.ndarray],
        next_stems: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray] | None:
        """Extract per-stem curves for a single transition.

        Args:
            mix_stems: dict of stem_name -> 1D audio array for the mix transition region
            prev_stems: dict of stem_name -> 1D audio array for previous track (time-aligned)
            next_stems: dict of stem_name -> 1D audio array for next track (time-aligned)

        Returns:
            dict mapping "{stem}_{param}" -> 1D curve array, or None if optimization fails.
            Keys like: drums_fader_prev, drums_fader_next, drums_eq_prev_low, etc.
        """
        all_results = {}

        for stem in config.STEMS:
            if stem not in mix_stems or stem not in prev_stems or stem not in next_stems:
                log.warning(f"Missing stem {stem}, skipping")
                continue

            S_dj = compute_spectrogram(mix_stems[stem], self.opt_config)
            S_prev = compute_spectrogram(prev_stems[stem], self.opt_config)
            S_next = compute_spectrogram(next_stems[stem], self.opt_config)

            # Ensure consistent frame count (use minimum)
            min_frames = min(S_dj.shape[1], S_prev.shape[1], S_next.shape[1])
            if min_frames < 2:
                log.warning(f"Stem {stem} has too few frames ({min_frames})")
                continue

            S_dj = S_dj[:, :min_frames]
            S_prev = S_prev[:, :min_frames]
            S_next = S_next[:, :min_frames]

            try:
                stem_result = self.optimizer.optimize(S_dj, S_prev, S_next)
            except Exception as e:
                log.warning(f"Optimization failed for stem {stem}: {e}")
                continue

            if stem_result is None:
                continue

            for key, val in stem_result.items():
                all_results[f"{stem}_{key}"] = val

        if not all_results:
            return None

        return all_results

    def save_curves(self, curves: dict, output_path: str | Path):
        """Save extracted curves to npz file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(output_path), **curves)

    def load_curves(self, path: str | Path) -> dict[str, np.ndarray] | None:
        """Load curves from npz file."""
        path = Path(path)
        if not path.exists():
            return None
        data = np.load(str(path))
        return dict(data)
