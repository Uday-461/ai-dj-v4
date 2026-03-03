#!/usr/bin/env python3
"""Align tracks to their mixes using DTW."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj.data.aligner import align_mix
from aidj import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--data-root", type=str, default=str(config.DATA_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.manifest) as f:
        manifest = json.load(f)

    mixes = manifest["mixes"]
    if args.limit:
        mixes = mixes[:args.limit]

    print(f"Aligning tracks for {len(mixes)} mixes...")

    total_alignments = 0
    good_alignments = 0

    for i, mix in enumerate(mixes):
        try:
            results = align_mix(mix, data_root=args.data_root)
            total_alignments += len(results)
            good_alignments += sum(1 for r in results if r.get('match_rate', 0) > 0.5)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(mixes)}] "
                      f"{total_alignments} alignments, "
                      f"{good_alignments} good (>0.5 match rate)")
        except Exception as e:
            logging.error(f"Error aligning mix {mix['id']}: {e}")

    print(f"\nAlignment complete:")
    print(f"  Total alignments: {total_alignments}")
    print(f"  Good alignments (>0.5): {good_alignments}")
    if total_alignments > 0:
        print(f"  Good rate: {good_alignments/total_alignments:.1%}")


if __name__ == "__main__":
    main()
