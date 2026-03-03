#!/usr/bin/env python3
"""Detect beats for all downloaded audio files."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj.data.beat_detector import BeatDetector
from aidj import config

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--data-root", type=str, default=str(config.DATA_ROOT))
    parser.add_argument("--method", choices=["beatnet", "librosa"], default="beatnet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-mix-audio", action="store_true",
                        help="Skip beat detection on full mix files (saves ~600MB RAM)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.manifest) as f:
        manifest = json.load(f)

    data_root = Path(args.data_root)
    beats_dir = data_root / "beats"
    beats_dir.mkdir(parents=True, exist_ok=True)

    detector = BeatDetector(method=args.method)

    # Collect all audio files to process
    audio_files = []
    mixes = manifest["mixes"]
    if args.limit:
        mixes = mixes[:args.limit]

    for mix in mixes:
        if not args.skip_mix_audio:
            mix_path = data_root / "mixes" / f"{mix['id']}.mp3"
            if mix_path.exists():
                audio_files.append(("mix", mix["id"], str(mix_path)))

        for track in mix.get("tracklist", []):
            tid = track.get("id")
            if tid:
                track_path = data_root / "tracks" / f"{tid}.mp3"
                if track_path.exists():
                    audio_files.append(("track", tid, str(track_path)))

    # Deduplicate by ID
    seen = set()
    unique_files = []
    for ftype, fid, fpath in audio_files:
        key = f"{ftype}_{fid}"
        if key not in seen:
            seen.add(key)
            unique_files.append((ftype, fid, fpath))

    print(f"Processing {len(unique_files)} audio files...")

    success = 0
    for i, (ftype, fid, fpath) in enumerate(unique_files):
        cache_path = beats_dir / f"{fid}.npz"
        if cache_path.exists():
            success += 1
            continue

        try:
            beats, downbeat_times = detector.corrected_beats(fpath)
            np.savez(cache_path, beats=beats,
                     downbeat_times=downbeat_times)
            success += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(unique_files)}] {success} succeeded")
        except Exception as e:
            log.warning(f"Beat detection failed for {fid}: {e}")

    print(f"\nBeat detection complete: {success}/{len(unique_files)} succeeded")


if __name__ == "__main__":
    main()
