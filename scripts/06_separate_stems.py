#!/usr/bin/env python3
"""Separate audio files into stems using Demucs."""
import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj.stems.separator import StemSeparator
from aidj.stems.stem_cache import StemCache
from aidj import config


def separate_tracks_and_mixes(mixes, data_root, separator, cache, skip_mixes):
    """Separate full track and mix audio files."""
    audio_files = []

    if not skip_mixes:
        for mix in mixes:
            mix_path = data_root / "mixes" / f"{mix['id']}.mp3"
            if mix_path.exists() and not cache.has_stems("mixes", mix["id"]):
                audio_files.append(("mixes", mix["id"], str(mix_path)))

    # Collect tracks (deduplicated)
    seen_tracks = set()
    for mix in mixes:
        for track in mix.get("tracklist", []):
            tid = track.get("id")
            if tid and tid not in seen_tracks:
                seen_tracks.add(tid)
                track_path = data_root / "tracks" / f"{tid}.mp3"
                if track_path.exists() and not cache.has_stems("tracks", tid):
                    audio_files.append(("tracks", tid, str(track_path)))

    print(f"Separating {len(audio_files)} audio files into stems...")

    success = 0
    for i, (atype, aid, apath) in enumerate(audio_files):
        try:
            cache.separate_and_cache(separator, apath, atype, aid)
            success += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(audio_files)}] {success} completed")
        except Exception as e:
            logging.error(f"Separation failed for {atype}/{aid}: {e}")

    print(f"\nStem separation complete: {success}/{len(audio_files)}")


def separate_mix_segments(mixes, data_root, separator, cache):
    """Separate only the transition regions of mixes into stems.

    For each transition, extracts the mix audio segment covering the
    overlap region and separates it with Demucs. This saves ~95% of
    compute vs separating full mixes.
    """
    total = 0
    success = 0
    skipped = 0

    for mix in mixes:
        mix_id = mix["id"]
        mix_path = data_root / "mixes" / f"{mix_id}.mp3"
        if not mix_path.exists():
            continue

        # Load transitions
        tran_path = data_root / "results" / "transitions" / f"{mix_id}.pkl"
        if not tran_path.exists():
            continue

        with open(tran_path, 'rb') as f:
            transitions = pickle.load(f)

        # Load mix audio once per mix (only if we have transitions to process)
        mix_audio = None

        for tran in transitions:
            tran_id = tran["tran_id"]
            total += 1

            if cache.has_stems("mix_segments", tran_id):
                skipped += 1
                continue

            # Lazy-load mix audio
            if mix_audio is None:
                import librosa
                mix_audio, _ = librosa.load(str(mix_path), sr=config.SR, mono=True)

            # Get transition region
            mix_cue_in_next = tran.get("mix_cue_in_time_next")
            mix_cue_out_prev = tran.get("mix_cue_out_time_prev")
            if mix_cue_in_next is None or mix_cue_out_prev is None:
                continue

            mix_start = min(mix_cue_in_next, mix_cue_out_prev)
            mix_end = max(mix_cue_in_next, mix_cue_out_prev)
            start_sample = int(mix_start * config.SR)
            end_sample = int(mix_end * config.SR)

            if end_sample - start_sample < config.SR:
                continue

            segment = mix_audio[start_sample:end_sample]

            try:
                cache.separate_and_cache_segment(
                    separator, segment, "mix_segments", tran_id,
                )
                success += 1
                if success % 10 == 0:
                    print(f"  [{total}] {success} segments separated, {skipped} cached")
            except Exception as e:
                logging.error(f"Segment separation failed for {tran_id}: {e}")

    print(f"\nMix segment separation complete: {success} new, "
          f"{skipped} cached, {total} total")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--data-root", type=str, default=str(config.DATA_ROOT))
    parser.add_argument("--device", type=str, default=None,
                        help="Device for Demucs (cuda, mps, cpu)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-mixes", action="store_true",
                        help="Skip mix separation (only separate tracks)")
    parser.add_argument("--include-mix-segments", action="store_true",
                        help="Separate transition regions of mixes for residual computation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.manifest) as f:
        manifest = json.load(f)

    data_root = Path(args.data_root)
    separator = StemSeparator(device=args.device)
    cache = StemCache(data_root)

    mixes = manifest["mixes"]
    if args.limit:
        mixes = mixes[:args.limit]

    if args.include_mix_segments:
        separate_mix_segments(mixes, data_root, separator, cache)
    else:
        separate_tracks_and_mixes(mixes, data_root, separator, cache, args.skip_mixes)


if __name__ == "__main__":
    main()
