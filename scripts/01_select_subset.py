#!/usr/bin/env python3
"""Select a subset of mixes from the djmix-dataset for training."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj.data.subset_selector import load_dataset, select_subset, save_manifest, print_stats
from aidj import config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=str(config.DATASET_JSON),
                        help="Path to djmix-dataset.json")
    parser.add_argument("--output", type=str, default="data/manifest.json",
                        help="Output manifest path")
    parser.add_argument("--size", type=int, default=config.SUBSET_SIZE,
                        help="Number of mixes to select")
    parser.add_argument("--min-transitions", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} mixes from dataset")

    subset = select_subset(dataset, size=args.size,
                          min_transitions=args.min_transitions,
                          min_track_coverage=args.min_coverage)

    manifest = save_manifest(subset, args.output)
    print_stats(manifest)

if __name__ == "__main__":
    main()
