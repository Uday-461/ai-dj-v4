#!/usr/bin/env python3
"""Extract transitions from aligned mixes."""
import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj.data.aligner import align_mix
from aidj.data.transition_extractor import extract_transitions
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

    print(f"Extracting transitions for {len(mixes)} mixes...")

    total_transitions = 0

    for i, mix in enumerate(mixes):
        try:
            alignment_results = align_mix(mix, data_root=args.data_root)
            transitions = extract_transitions(mix, alignment_results, data_root=args.data_root)
            total_transitions += len(transitions)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(mixes)}] {total_transitions} transitions extracted")
        except Exception as e:
            logging.error(f"Error extracting transitions for {mix['id']}: {e}")

    print(f"\nTransition extraction complete:")
    print(f"  Total transitions: {total_transitions}")
    if len(mixes) > 0:
        print(f"  Average per mix: {total_transitions/len(mixes):.1f}")


if __name__ == "__main__":
    main()
