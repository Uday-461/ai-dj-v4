#!/usr/bin/env python3
"""Create comprehensive training manifest including ALL available mixes/residuals.

Queries HuggingFace Hub for all mixes with residuals (even partial data),
collects all usable transitions, and creates a manifest suitable for parallel
CPU processing and resumable curve extraction.

Usage:
    python scripts/create_training_manifest.py
    # Output: data/manifest_training_all_available.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj import config

log = logging.getLogger(__name__)


def query_hf_residuals(hf_token: str = None) -> dict:
    """Query HuggingFace for all mixes with residuals.

    Returns dict mapping mix_id -> {residuals: [...], stems_available: bool}
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("huggingface_hub not installed. Install with: pip install huggingface-hub")
        return {}

    api = HfApi()
    repo_id = "Uday-4/djmix-v3"

    log.info(f"Querying HuggingFace Hub for residuals in {repo_id}...")

    mixes_with_residuals = {}

    try:
        # List all files in the repo
        files = api.list_repo_tree(repo_id, token=hf_token, recursive=True)

        # Track mixes that have residuals
        for file_info in files:
            path = file_info.path

            # Check for residuals: results/residuals/{mix_id}/{residual_id}.npy
            if path.startswith("results/residuals/"):
                parts = path.split("/")
                if len(parts) >= 3:
                    mix_id = parts[2]
                    if mix_id not in mixes_with_residuals:
                        mixes_with_residuals[mix_id] = {
                            "residuals": [],
                            "stems_available": False
                        }
                    mixes_with_residuals[mix_id]["residuals"].append(path)

            # Check for stems: stems/tracks/{track_id}/... or stems/mixes/...
            if path.startswith("stems/tracks/") or path.startswith("stems/mix_segments/"):
                # Mark that stems are available
                for mix_id in mixes_with_residuals:
                    mixes_with_residuals[mix_id]["stems_available"] = True

        log.info(f"Found {len(mixes_with_residuals)} mixes with residuals")
        return mixes_with_residuals

    except Exception as e:
        log.error(f"Failed to query HuggingFace: {e}")
        return {}


def load_transitions_manifest(manifest_path: str | Path) -> Optional[dict]:
    """Load transitions from existing manifest (if available).

    Returns dict mapping mix_id -> list of transitions.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        log.warning(f"Manifest not found: {manifest_path}")
        return None

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        transitions_by_mix = {}
        for mix in manifest.get("mixes", []):
            mix_id = mix["id"]
            transitions_by_mix[mix_id] = {
                "num_available_transitions": mix.get("num_available_transitions", 0),
                "title": mix.get("title", ""),
                "has_complete_stems": len(mix.get("tracklist", [])) > 0,
            }

        return transitions_by_mix
    except Exception as e:
        log.error(f"Failed to load manifest: {e}")
        return None


def create_training_manifest(
    output_path: str | Path = "data/manifest_training_all_available.json",
    reference_manifest: str | Path = "data/manifest_100mix.json",
    hf_token: Optional[str] = None,
) -> dict:
    """Create comprehensive training manifest.

    Args:
        output_path: Where to save the manifest
        reference_manifest: Existing manifest to extract mix info from
        hf_token: HuggingFace token (optional)

    Returns:
        The created manifest dict
    """
    output_path = Path(output_path)
    reference_manifest = Path(reference_manifest)

    # Load reference manifest for mix info
    log.info(f"Loading reference manifest: {reference_manifest}")
    transitions_by_mix = load_transitions_manifest(reference_manifest)
    if not transitions_by_mix:
        transitions_by_mix = {}

    # Query HF for actual residuals available
    log.info("Querying HuggingFace for available residuals...")
    hf_residuals = query_hf_residuals(hf_token)

    # Build training manifest
    manifest_mixes = []
    total_transitions = 0
    total_residuals = 0
    mixes_with_partial_data = 0

    for mix_id, residual_info in sorted(hf_residuals.items()):
        num_residuals = len(residual_info["residuals"])
        total_residuals += num_residuals

        # Get transition count from reference manifest if available
        ref_info = transitions_by_mix.get(mix_id, {})
        num_transitions = ref_info.get("num_available_transitions", num_residuals)
        total_transitions += num_transitions

        # Determine if usable: has residuals + any stem data available
        is_usable = num_residuals > 0 and residual_info["stems_available"]

        mix_entry = {
            "id": mix_id,
            "usable_transitions": num_transitions,
            "num_residuals": num_residuals,
            "has_complete_stems": ref_info.get("has_complete_stems", False),
            "title": ref_info.get("title", ""),
            "notes": "partial residuals" if num_residuals > 0 and not ref_info.get("has_complete_stems", False) else "",
        }

        if is_usable:
            manifest_mixes.append(mix_entry)
            if num_residuals < num_transitions:
                mixes_with_partial_data += 1

    # Build final manifest
    manifest = {
        "metadata": {
            "total_mixes": len(manifest_mixes),
            "total_transitions": total_transitions,
            "total_residuals": total_residuals,
            "mixes_with_partial_data": mixes_with_partial_data,
            "creation_date": datetime.now().isoformat(),
            "includes_partial_data": True,
            "source": "HuggingFace Hub (Uday-4/djmix-v3)",
            "description": "Comprehensive training manifest for all available mixes/residuals"
        },
        "mixes": manifest_mixes
    }

    # Save manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Manifest saved to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"Training Manifest Created")
    print(f"{'='*70}")
    print(f"Total mixes:              {len(manifest_mixes)}")
    print(f"Total transitions:        {total_transitions}")
    print(f"Total residuals:          {total_residuals}")
    print(f"Mixes with partial data:  {mixes_with_partial_data}")
    print(f"Creation date:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output:                   {output_path}")
    print(f"{'='*70}\n")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/manifest_training_all_available.json",
        help="Output path for training manifest"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/manifest_100mix.json",
        help="Reference manifest to extract mix info from"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, uses HF_TOKEN env var if not provided)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    # Get HF token from env if not provided
    hf_token = args.hf_token or __import__("os").environ.get("HF_TOKEN")

    create_training_manifest(
        output_path=args.output,
        reference_manifest=args.reference,
        hf_token=hf_token
    )


if __name__ == "__main__":
    main()
