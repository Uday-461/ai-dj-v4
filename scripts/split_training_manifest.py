#!/usr/bin/env python3
"""Split training manifest into 2 equal parts for parallel CPU processing.

Splits mixes 50/50 by count for even distribution of work across 2 machines.
Creates balanced part1 and part2 manifests that can be processed in parallel.

Usage:
    python scripts/split_training_manifest.py data/manifest_training_all_available.json
    # Output: data/manifest_training_part1.json, data/manifest_training_part2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def split_manifest_balanced(manifest: dict, n_parts: int = 2) -> list:
    """Split mixes into n_parts using round-robin assignment for balance.

    This ensures that if mixes are sorted by transitions (descending),
    each part gets a roughly equal distribution of work.
    """
    mixes = manifest.get("mixes", [])

    # Sort by transitions descending for better load balancing
    sorted_mixes = sorted(
        mixes,
        key=lambda m: m.get("usable_transitions", 0),
        reverse=True
    )

    # Round-robin assignment
    parts = [[] for _ in range(n_parts)]
    for i, mix in enumerate(sorted_mixes):
        parts[i % n_parts].append(mix)

    return parts


def build_part_manifest(mixes: list, part_num: int) -> dict:
    """Build a manifest dict for a part."""
    metadata = {
        "part": part_num,
        "num_mixes": len(mixes),
        "num_transitions": sum(m.get("usable_transitions", 0) for m in mixes),
        "num_residuals": sum(m.get("num_residuals", 0) for m in mixes),
    }

    return {
        "metadata": metadata,
        "mixes": mixes
    }


def split_training_manifest(
    manifest_path: str | Path,
    output_dir: Optional[str | Path] = None,
    n_parts: int = 2
) -> list:
    """Split manifest into parts and save.

    Returns list of output paths.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir or manifest_path.parent)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Split mixes
    parts = split_manifest_balanced(manifest, n_parts)

    # Save parts
    stem = manifest_path.stem
    output_paths = []

    for i, mixes in enumerate(parts, 1):
        part_manifest = build_part_manifest(mixes, i)
        out_path = output_dir / f"{stem}_part{i}.json"

        with open(out_path, "w") as f:
            json.dump(part_manifest, f, indent=2)

        output_paths.append(out_path)

        # Print summary
        meta = part_manifest["metadata"]
        print(
            f"Part {i}: {meta['num_mixes']:3d} mixes | "
            f"{meta['num_transitions']:4d} transitions | "
            f"{meta['num_residuals']:4d} residuals | {out_path}"
        )

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "manifest",
        type=str,
        help="Path to training manifest (e.g., manifest_training_all_available.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=2,
        help="Number of parts to split into (default: 2)"
    )
    args = parser.parse_args()

    print(f"\nSplitting manifest: {args.manifest}")
    print(f"{'='*80}")

    split_training_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        n_parts=args.parts
    )

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
