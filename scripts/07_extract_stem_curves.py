#!/usr/bin/env python3
"""Extract per-stem fader + EQ curves for all transitions.

Supports multiprocessing for parallel curve extraction across transitions.
"""
import argparse
import json
import logging
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj.curves.stem_curve_extractor import StemCurveExtractor
from aidj.curves.optimizer import OptConfig
from aidj.stems.stem_cache import StemCache
from aidj import config

log = logging.getLogger(__name__)


def _extract_one(args):
    """Worker function for multiprocessing. Extracts curves for one transition."""
    tran, data_root, output_path = args

    tran_id = tran["tran_id"]
    prev_id = tran["track_id_prev"]
    next_id = tran["track_id_next"]

    stem_cache = StemCache(data_root)

    mix_seg_stems = stem_cache.load_stems("mix_segments", tran_id)
    prev_stems = stem_cache.load_stems("tracks", prev_id)
    next_stems = stem_cache.load_stems("tracks", next_id)

    if mix_seg_stems is None or prev_stems is None or next_stems is None:
        return tran_id, False, "missing stems"

    try:
        region_len = min(len(v) for v in mix_seg_stems.values())

        # Cap transition length to avoid slow CVXPY solves
        max_samples = int(config.MAX_TRANSITION_SECS * config.OPT_SR)
        if region_len > max_samples:
            region_len = max_samples

        if region_len < config.OPT_SR:
            return tran_id, False, "too short"

        mix_region = {s: mix_seg_stems[s][:region_len] for s in config.STEMS}

        track_start_prev = tran.get("track_cue_in_time_prev", 0)
        track_start_next = tran.get("track_cue_in_time_next", 0)

        prev_start = int(track_start_prev * config.OPT_SR)
        prev_region = {s: prev_stems[s][prev_start:prev_start + region_len] for s in config.STEMS}

        next_start = int(track_start_next * config.OPT_SR)
        next_region = {s: next_stems[s][next_start:next_start + region_len] for s in config.STEMS}

        min_len = min(
            region_len,
            min(len(v) for v in prev_region.values()),
            min(len(v) for v in next_region.values()),
        )
        if min_len < config.OPT_SR:
            return tran_id, False, "too short after trim"

        mix_region = {s: v[:min_len] for s, v in mix_region.items()}
        prev_region = {s: v[:min_len] for s, v in prev_region.items()}
        next_region = {s: v[:min_len] for s, v in next_region.items()}

        extractor = StemCurveExtractor()
        curves = extractor.extract_transition_curves(mix_region, prev_region, next_region)

        if curves is not None:
            extractor.save_curves(curves, output_path)
            return tran_id, True, None
        else:
            return tran_id, False, "optimization returned None"
    except Exception as e:
        return tran_id, False, str(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--data-root", type=str, default=str(config.DATA_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count / 2)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    data_root = Path(args.data_root)

    with open(args.manifest) as f:
        manifest = json.load(f)

    curves_dir = data_root / "results" / "stem_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    mixes = manifest["mixes"]
    if args.limit:
        mixes = mixes[:args.limit]

    # Collect all work items
    work_items = []
    already_done = 0

    for mix in mixes:
        mix_id = mix["id"]

        tran_path = data_root / "results" / "transitions" / f"{mix_id}.pkl"
        if not tran_path.exists():
            continue

        with open(tran_path, 'rb') as f:
            transitions = pickle.load(f)

        for tran in transitions:
            tran_id = tran["tran_id"]
            output_path = curves_dir / mix_id / f"{tran_id}.npz"

            if output_path.exists():
                already_done += 1
                continue

            work_items.append((tran, str(data_root), str(output_path)))

    log.info(f"Curve extraction: {len(work_items)} pending, {already_done} already done")

    if not work_items:
        print(f"\nCurve extraction complete: {already_done} curves (all cached)")
        return

    n_workers = args.workers or max(1, (os.cpu_count() or 2) // 2)
    total_success = already_done
    total_failed = 0

    if n_workers == 1:
        # Sequential mode
        for i, item in enumerate(work_items):
            tran_id, success, err = _extract_one(item)
            if success:
                total_success += 1
                log.info(f"[{i+1}/{len(work_items)}] {tran_id}: OK")
            else:
                total_failed += 1
                log.warning(f"[{i+1}/{len(work_items)}] {tran_id}: {err}")
    else:
        log.info(f"Using {n_workers} parallel workers")
        with Pool(n_workers) as pool:
            for i, (tran_id, success, err) in enumerate(pool.imap_unordered(_extract_one, work_items)):
                if success:
                    total_success += 1
                    log.info(f"[{i+1}/{len(work_items)}] {tran_id}: OK")
                else:
                    total_failed += 1
                    log.warning(f"[{i+1}/{len(work_items)}] {tran_id}: {err}")

    print(f"\nCurve extraction complete:")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failed}")


if __name__ == "__main__":
    main()
