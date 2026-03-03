from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aidj import config

log = logging.getLogger(__name__)


class StemTransitionDataset(Dataset):
    """PyTorch Dataset for stem transition training.

    Each sample contains:
    - input: (16, n_mels, n_frames) — per-stem spectrograms:
        4 stems x 4 channels (prev, next, prev_residual, next_residual)
    - target: (4, 8, n_frames) — per-stem fader + EQ curves
    """

    def __init__(self, data_dir: str | Path, n_mels: int = 128,
                 max_frames: int = 256):
        self.data_dir = Path(data_dir)
        self.n_mels = n_mels
        self.max_frames = max_frames

        manifest_path = self.data_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.samples = manifest["samples"]
        log.info(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        curves_path = sample["curves_path"]

        # Load curves
        curves_data = np.load(curves_path)

        # Build target tensor: (4, 8, n_frames)
        # For each stem: [fader_prev, fader_next, eq_prev_low, eq_prev_mid, eq_prev_high,
        #                  eq_next_low, eq_next_mid, eq_next_high]
        targets = []
        n_frames = None

        for stem in config.STEMS:
            stem_curves = []
            for param in ['fader_prev', 'fader_next',
                         'eq_prev_low', 'eq_prev_mid', 'eq_prev_high',
                         'eq_next_low', 'eq_next_mid', 'eq_next_high']:
                key = f"{stem}_{param}"
                if key in curves_data:
                    curve = curves_data[key]
                    if n_frames is None:
                        n_frames = len(curve)
                    stem_curves.append(curve[:n_frames])
                else:
                    if n_frames is None:
                        n_frames = self.max_frames
                    stem_curves.append(np.zeros(n_frames))
            targets.append(np.stack(stem_curves))

        target = np.stack(targets)  # (4, 8, n_frames)

        # Pad or truncate to max_frames
        if n_frames > self.max_frames:
            target = target[:, :, :self.max_frames]
            n_frames = self.max_frames
        elif n_frames < self.max_frames:
            pad_width = ((0, 0), (0, 0), (0, self.max_frames - n_frames))
            target = np.pad(target, pad_width, mode='edge')

        # Build input: load or compute per-stem spectrograms
        # For now, create placeholder spectrograms from metadata
        # In practice, these would be pre-computed and stored alongside curves
        input_tensor = self._load_or_compute_input(sample, n_frames)

        return (
            torch.from_numpy(input_tensor).float(),
            torch.from_numpy(target).float(),
        )

    def _load_or_compute_input(self, sample, n_frames):
        """Load pre-computed input spectrograms or create placeholder."""
        from aidj import config as cfg
        C = cfg.MODEL_IN_CHANNELS  # 4
        n_channels = cfg.MODEL_N_STEMS * C  # 16

        # Check if pre-computed input exists
        curves_path = Path(sample["curves_path"])
        input_path = curves_path.with_suffix('.input.npz')

        if input_path.exists():
            data = np.load(str(input_path))
            specs = data['spectrograms']  # (16, n_mels, T) or old (8, n_mels, T)

            if specs.shape[0] == n_channels:
                pass  # new 16-channel format, use as-is
            elif specs.shape[0] == cfg.MODEL_N_STEMS * 2:
                # Old 8-channel format: [s0_prev, s0_next, s1_prev, s1_next, ...]
                # Expand to 16 by inserting zero residual channels per stem
                new_specs = np.zeros(
                    (n_channels, specs.shape[1], specs.shape[2]),
                    dtype=specs.dtype,
                )
                for s in range(cfg.MODEL_N_STEMS):
                    new_specs[s * C]     = specs[s * 2]      # prev
                    new_specs[s * C + 1] = specs[s * 2 + 1]  # next
                    # channels s*C+2 and s*C+3 stay zero (residuals)
                specs = new_specs
            else:
                # Unknown format, zero-pad
                pad = np.zeros(
                    (n_channels - specs.shape[0], specs.shape[1], specs.shape[2]),
                    dtype=specs.dtype,
                )
                specs = np.concatenate([specs, pad], axis=0)

            if specs.shape[2] > self.max_frames:
                specs = specs[:, :, :self.max_frames]
            elif specs.shape[2] < self.max_frames:
                pad_width = ((0, 0), (0, 0), (0, self.max_frames - specs.shape[2]))
                specs = np.pad(specs, pad_width, mode='constant')
            return specs

        # Placeholder: zeros
        return np.zeros((n_channels, self.n_mels, self.max_frames), dtype=np.float32)
