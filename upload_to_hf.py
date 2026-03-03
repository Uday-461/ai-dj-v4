#!/usr/bin/env python3
"""Upload djmix data to HuggingFace Hub.

Uses upload_large_folder for progress tracking and resume support.

Usage:
    python upload_to_hf.py --source ~/djmix
    python upload_to_hf.py --source ~/djmix --repo Uday-4/djmix-v3
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_REPO = "Uday-4/djmix-v3"

IGNORE_PATTERNS = [
    "**/*.wav",          # exclude WAV stems (OGG replaces them)
    "_tmp_*.json",       # temp manifests
    "download_progress.json",
    "pipeline_progress.json",
]


def main():
    parser = argparse.ArgumentParser(description="Upload djmix data to HuggingFace Hub")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO,
                        help=f"HF dataset repo ID (default: {DEFAULT_REPO})")
    parser.add_argument("--source", type=str, required=True,
                        help="Source directory (AIDJ_DATA_ROOT)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set. Add it to v3/.env or export it.")

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=True, exist_ok=True)

    print(f"Uploading {source} -> hf://datasets/{args.repo}")
    print(f"Excluding: {IGNORE_PATTERNS}")

    api.upload_large_folder(
        folder_path=str(source),
        repo_id=args.repo,
        repo_type="dataset",
        ignore_patterns=IGNORE_PATTERNS,
    )

    print(f"\nDone! View at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
