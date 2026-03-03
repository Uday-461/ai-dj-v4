#!/usr/bin/env python3
"""Compute mix residuals: difference between DJ mix stems and original track stems.

For each transition, loads the mix segment stems and original track stems,
aligns them, subtracts, and saves per-stem residual spectrograms.

Requires step 06 to have been run with --include-mix-segments.
"""
import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj import config
from aidj.stems.stem_cache import StemCache
from aidj.data.residual import compute_residual, align_track_to_mix_segment

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--data-root", type=str, default=str(config.DATA_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.manifest) as f:
        manifest = json.load(f)

    data_root = Path(args.data_root)
    stem_cache = StemCache(data_root)

    residuals_dir = data_root / "results" / "residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)

    mixes = manifest["mixes"]
    if args.limit:
        mixes = mixes[:args.limit]

    success = 0
    skipped = 0
    total = 0

    for mix in mixes:
        mix_id = mix["id"]

        # Load transitions
        tran_path = data_root / "results" / "transitions" / f"{mix_id}.pkl"
        if not tran_path.exists():
            continue

        with open(tran_path, 'rb') as f:
            transitions = pickle.load(f)

        mix_res_dir = residuals_dir / mix_id
        mix_res_dir.mkdir(parents=True, exist_ok=True)

        for tran in transitions:
            tran_id = tran["tran_id"]
            total += 1

            out_path = mix_res_dir / f"{tran_id}.npz"
            if out_path.exists():
                skipped += 1
                continue

            # Get transition region times
            mix_cue_in_next = tran.get("mix_cue_in_time_next")
            mix_cue_out_prev = tran.get("mix_cue_out_time_prev")
            if mix_cue_in_next is None or mix_cue_out_prev is None:
                continue

            # The transition region in the mix
            mix_start = min(mix_cue_in_next, mix_cue_out_prev)
            mix_end = max(mix_cue_in_next, mix_cue_out_prev)
            region_len = int((mix_end - mix_start) * config.SR)
            if region_len < config.SR:
                continue

            prev_id = tran["track_id_prev"]
            next_id = tran["track_id_next"]

            # Load mix segment stems
            mix_seg_stems = stem_cache.load_stems("mix_segments", tran_id)

            # Load track stems
            prev_track_stems = stem_cache.load_stems("tracks", prev_id)
            next_track_stems = stem_cache.load_stems("tracks", next_id)

            if mix_seg_stems is None:
                log.debug(f"No mix segment stems for {tran_id}")
                continue

            residual_data = {}

            # Compute residual for prev track
            if prev_track_stems is not None:
                track_start_prev = tran.get("track_cue_in_time_prev", 0)
                # Align prev track audio to mix segment
                aligned_prev = {}
                for stem in config.STEMS:
                    if stem in prev_track_stems:
                        aligned_prev[stem] = align_track_to_mix_segment(
                            prev_track_stems[stem], track_start_prev,
                            region_len, config.SR,
                        )

                prev_residuals = compute_residual(mix_seg_stems, aligned_prev)
                for stem, spec in prev_residuals.items():
                    residual_data[f"{stem}_prev"] = spec

            # Compute residual for next track
            if next_track_stems is not None:
                track_start_next = tran.get("track_cue_in_time_next", 0)
                aligned_next = {}
                for stem in config.STEMS:
                    if stem in next_track_stems:
                        aligned_next[stem] = align_track_to_mix_segment(
                            next_track_stems[stem], track_start_next,
                            region_len, config.SR,
                        )

                next_residuals = compute_residual(mix_seg_stems, aligned_next)
                for stem, spec in next_residuals.items():
                    residual_data[f"{stem}_next"] = spec

            if residual_data:
                np.savez_compressed(str(out_path), **residual_data)
                success += 1

            if (total) % 20 == 0:
                print(f"  [{total}] {success} computed, {skipped} cached")

    print(f"\nResidual computation complete: {success} computed, "
          f"{skipped} cached, {total} total")


if __name__ == "__main__":
    main()
