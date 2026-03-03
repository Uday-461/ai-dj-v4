#!/usr/bin/env python3
"""Evaluate a trained StemTransitionNet model."""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj.model.architecture import StemTransitionNet
from aidj.model.dataset import StemTransitionDataset
from aidj.model.losses import TransitionLoss
from aidj import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--test-dir", type=str, default="data/training/test")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model = StemTransitionNet.load(args.model, device=args.device)
    device = next(model.parameters()).device

    dataset = StemTransitionDataset(args.test_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    criterion = TransitionLoss()

    # Per-stem, per-param MAE tracking
    stem_param_mae = {stem: {param: [] for param in [
        'fader_prev', 'fader_next', 'eq_prev_low', 'eq_prev_mid',
        'eq_prev_high', 'eq_next_low', 'eq_next_mid', 'eq_next_high'
    ]} for stem in config.STEMS}

    total_mae = 0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            pred = model(inputs)
            _, loss_dict = criterion(pred, targets)
            total_mae += loss_dict['mae']
            n_batches += 1

            # Per-stem, per-param breakdown
            for s, stem in enumerate(config.STEMS):
                for p, param in enumerate(['fader_prev', 'fader_next',
                    'eq_prev_low', 'eq_prev_mid', 'eq_prev_high',
                    'eq_next_low', 'eq_next_mid', 'eq_next_high']):
                    mae_val = torch.mean(torch.abs(
                        pred[:, s, p, :] - targets[:, s, p, :]
                    )).item()
                    stem_param_mae[stem][param].append(mae_val)

    print(f"\nOverall test MAE: {total_mae / max(n_batches, 1):.6f}")
    print(f"\nPer-stem, per-param MAE:")
    print(f"{'Stem':<10} {'Param':<15} {'MAE':<10}")
    print("-" * 35)
    for stem in config.STEMS:
        for param, values in stem_param_mae[stem].items():
            mean_mae = np.mean(values) if values else 0
            print(f"{stem:<10} {param:<15} {mean_mae:.6f}")


if __name__ == "__main__":
    main()
