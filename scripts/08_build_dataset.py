#!/usr/bin/env python3
"""Build training dataset from extracted stem curves.

Pre-computes per-stem mel spectrograms (.input.npz) alongside curve files
so the dataset can load real inputs instead of zero placeholders.
"""
import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj.stems.stem_cache import StemCache
from aidj import config

log = logging.getLogger(__name__)


def compute_stem_spectrograms(stem_cache, tran, data_root, n_mels=128, max_frames=256):
    """Compute per-stem mel spectrograms for a transition's prev+next tracks,
    plus residual channels if available.

    Returns (16, n_mels, n_frames) array: for each of 4 stems, 4 channels:
      [prev_spec, next_spec, prev_residual, next_residual]
    Or None on failure.
    """
    prev_id = tran["track_id_prev"]
    next_id = tran["track_id_next"]
    tran_id = tran["tran_id"]
    mix_id = tran["mix_id"]

    prev_stems = stem_cache.load_stems("tracks", prev_id)
    next_stems = stem_cache.load_stems("tracks", next_id)

    if prev_stems is None or next_stems is None:
        return None

    # Get transition time region
    mix_start = tran.get("mix_cue_in_time_next")
    mix_end = tran.get("mix_cue_out_time_prev")
    if mix_start is None or mix_end is None:
        return None

    track_start_prev = tran.get("track_cue_in_time_prev", 0)
    track_start_next = tran.get("track_cue_in_time_next", 0)

    region_len = int(abs(mix_end - mix_start) * config.SR)
    if region_len < config.SR:
        return None

    # Load residuals if available
    residuals_path = Path(data_root) / "results" / "residuals" / mix_id / f"{tran_id}.npz"
    residual_data = None
    if residuals_path.exists():
        residual_data = np.load(str(residuals_path))

    channels = []
    for stem in config.STEMS:
        # Channel 0: prev track spectrogram
        # Channel 1: next track spectrogram
        for stems_dict, start_time in [(prev_stems, track_start_prev),
                                        (next_stems, track_start_next)]:
            audio = stems_dict[stem]
            start_sample = int(start_time * config.SR)
            segment = audio[start_sample:start_sample + region_len]
            if len(segment) < region_len:
                segment = np.pad(segment, (0, region_len - len(segment)))

            mel = librosa.feature.melspectrogram(
                y=segment.astype(np.float32), sr=config.SR,
                n_fft=config.OPT_FFT, hop_length=config.OPT_HOP,
                n_mels=n_mels, power=1,
            )
            mel_db = librosa.amplitude_to_db(mel, ref=np.max)
            mel_db = (mel_db + 80) / 80
            channels.append(mel_db)

        # Channel 2: prev residual spectrogram
        # Channel 3: next residual spectrogram
        # Reference shape from the track spectrograms just computed
        ref_shape = channels[-1].shape  # (n_mels, T)
        for suffix in ["_prev", "_next"]:
            key = f"{stem}{suffix}"
            if residual_data is not None and key in residual_data:
                res = residual_data[key]
                # Match frame count to track spectrograms
                if res.shape[1] > ref_shape[1]:
                    res = res[:, :ref_shape[1]]
                elif res.shape[1] < ref_shape[1]:
                    pad_w = ((0, 0), (0, ref_shape[1] - res.shape[1]))
                    res = np.pad(res, pad_w, mode='constant')
                channels.append(res)
            else:
                channels.append(np.zeros(ref_shape, dtype=np.float32))

    # Stack to (16, n_mels, n_frames), pad/truncate to max_frames
    specs = np.stack(channels)  # (16, n_mels, T)
    T = specs.shape[2]
    if T > max_frames:
        specs = specs[:, :, :max_frames]
    elif T < max_frames:
        pad_width = ((0, 0), (0, 0), (0, max_frames - T))
        specs = np.pad(specs, pad_width, mode='constant')

    return specs.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--data-root", type=str, default=str(config.DATA_ROOT))
    parser.add_argument("--output-dir", type=str, default="data/training")
    parser.add_argument("--min-match-rate", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=256)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    curves_dir = data_root / "results" / "stem_curves"
    stem_cache = StemCache(data_root)

    with open(args.manifest) as f:
        manifest = json.load(f)

    # Collect all valid samples
    samples = []
    n_specs_computed = 0

    n_mixes = len(manifest["mixes"])
    for mix_idx, mix in enumerate(manifest["mixes"]):
        mix_id = mix["id"]

        tran_path = data_root / "results" / "transitions" / f"{mix_id}.pkl"
        if not tran_path.exists():
            log.info(f"[{mix_idx+1}/{n_mixes}] {mix_id}: no transitions PKL, skipping")
            continue

        with open(tran_path, 'rb') as f:
            transitions = pickle.load(f)

        mix_specs = 0
        mix_skipped = 0
        mix_cached = 0

        for tran in transitions:
            tran_id = tran["tran_id"]

            # Quality filter
            min_mr = min(tran.get("match_rate_prev", 0), tran.get("match_rate_next", 0))
            if min_mr < args.min_match_rate:
                continue

            curves_path = curves_dir / mix_id / f"{tran_id}.npz"
            if not curves_path.exists():
                continue

            # Pre-compute input spectrograms if not already done
            input_path = curves_path.with_suffix('.input.npz')
            if not input_path.exists():
                specs = compute_stem_spectrograms(
                    stem_cache, tran, data_root,
                    n_mels=args.n_mels, max_frames=args.max_frames,
                )
                if specs is None:
                    log.warning(f"Could not compute spectrograms for {tran_id}")
                    mix_skipped += 1
                    continue
                np.savez_compressed(str(input_path), spectrograms=specs)
                n_specs_computed += 1
                mix_specs += 1
            else:
                mix_cached += 1

            samples.append({
                "tran_id": tran_id,
                "mix_id": mix_id,
                "track_id_prev": tran["track_id_prev"],
                "track_id_next": tran["track_id_next"],
                "curves_path": str(curves_path),
                "match_rate_prev": tran.get("match_rate_prev", 0),
                "match_rate_next": tran.get("match_rate_next", 0),
            })

        log.info(f"[{mix_idx+1}/{n_mixes}] {mix_id}: {mix_specs} computed, {mix_cached} cached, {mix_skipped} skipped")

    print(f"Found {len(samples)} valid samples ({n_specs_computed} spectrograms computed)")

    if not samples:
        print("No samples found. Exiting.")
        return

    # Split
    np.random.seed(42)
    np.random.shuffle(samples)
    n_train = int(len(samples) * args.train_ratio)
    n_val = int(len(samples) * args.val_ratio)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split_name, split_samples in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        manifest_out = {
            "split": split_name,
            "num_samples": len(split_samples),
            "samples": split_samples,
        }
        with open(split_dir / "manifest.json", "w") as f:
            json.dump(manifest_out, f, indent=2)

        print(f"  {split_name}: {len(split_samples)} samples")

    print(f"\nDataset built in {output_dir}")


if __name__ == "__main__":
    main()
