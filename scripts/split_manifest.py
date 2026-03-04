#!/usr/bin/env python3
"""Split a manifest into N parts for parallel processing on multiple machines.

Uses alternating (round-robin) assignment so that mixes sorted by transition
count descending are balanced evenly across parts.

Usage:
    python scripts/split_manifest.py data/manifest_100mix.json --parts 2
    # Output: data/manifest_100mix_part1.json, data/manifest_100mix_part2.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def split_manifest(manifest, n_parts):
    """Split mixes into n_parts using round-robin assignment."""
    parts = [[] for _ in range(n_parts)]
    for i, mix in enumerate(manifest["mixes"]):
        parts[i % n_parts].append(mix)
    return parts


def build_manifest(mixes):
    """Build a manifest dict from a list of mixes."""
    return {
        "num_mixes": len(mixes),
        "num_tracks": sum(len(m.get("tracklist", [])) for m in mixes),
        "num_transitions": sum(m.get("num_available_transitions", 0) for m in mixes),
        "mixes": mixes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=str, help="Path to input manifest JSON")
    parser.add_argument("--parts", type=int, default=2, help="Number of parts to split into")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)

    parts = split_manifest(manifest, args.parts)

    stem = manifest_path.stem
    parent = manifest_path.parent

    for i, mixes in enumerate(parts, 1):
        part_manifest = build_manifest(mixes)
        out_path = parent / f"{stem}_part{i}.json"
        with open(out_path, "w") as f:
            json.dump(part_manifest, f, indent=2)
        print(f"Part {i}: {part_manifest['num_mixes']} mixes, "
              f"{part_manifest['num_tracks']} tracks, "
              f"{part_manifest['num_transitions']} transitions -> {out_path}")


if __name__ == "__main__":
    main()
