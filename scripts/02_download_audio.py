#!/usr/bin/env python3
"""Download audio for mixes and tracks using Apify (primary) + yt-dlp (fallback).

Apify actor: marielise.dev~youtube-video-downloader (~$0.015/track)
yt-dlp fallback: free but rate-limited.

Progress is tracked in DATA_ROOT/download_progress.json for resume support.

Usage:
    python scripts/02_download_audio.py --manifest data/manifest_2mix.json
    python scripts/02_download_audio.py --manifest data/manifest.json --apify-budget 5.0
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidj import config

log = logging.getLogger(__name__)

APIFY_ACTOR = "marielise.dev~youtube-video-downloader"
APIFY_POLL_INTERVAL = 5  # seconds
APIFY_TIMEOUT = 300  # seconds


def load_progress(progress_path):
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}, "method": {}}


def save_progress(progress, progress_path):
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def download_via_apify(youtube_url, output_path, apify_token):
    """Download audio via Apify YouTube actor. Returns True on success."""
    try:
        from apify_client import ApifyClient
    except ImportError:
        log.warning("apify-client not installed, skipping Apify")
        return False

    if not apify_token:
        return False

    client = ApifyClient(apify_token)

    run_input = {
        "urls": [{"url": youtube_url}],
        "format": "mp3",
    }

    try:
        run = client.actor(APIFY_ACTOR).call(
            run_input=run_input,
            timeout_secs=APIFY_TIMEOUT,
        )

        if run.get("status") != "SUCCEEDED":
            log.warning(f"Apify run failed with status: {run.get('status')}")
            return False

        # Get results from dataset
        dataset_id = run["defaultDatasetId"]
        items = list(client.dataset(dataset_id).iterate_items())

        if not items:
            log.warning("Apify returned no items")
            return False

        download_url = items[0].get("downloadUrl")
        if not download_url:
            log.warning("No downloadUrl in Apify result")
            return False

        # Download the MP3 file
        import urllib.request
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        urllib.request.urlretrieve(download_url, output_path)
        log.info(f"Apify downloaded: {output_path}")
        return True

    except Exception as e:
        log.warning(f"Apify download failed: {e}")
        return False


def download_via_ytdlp(youtube_url, output_path, limit_rate="5M"):
    """Download audio via yt-dlp with rate limiting. Returns True on success."""
    from aidj.data.downloader import download_audio
    return download_audio(youtube_url, output_path, limit_rate=limit_rate)


def download_track(track_id, data_root, apify_token, use_apify=True):
    """Download a single track, trying Apify first then yt-dlp.

    Returns (success, method) tuple.
    """
    output_path = str(Path(data_root) / "tracks" / f"{track_id}.mp3")

    if os.path.isfile(output_path):
        return True, "cached"

    youtube_url = f"https://www.youtube.com/watch?v={track_id}"

    # Try Apify first
    if use_apify and apify_token:
        if download_via_apify(youtube_url, output_path, apify_token):
            return True, "apify"

    # Fallback to yt-dlp
    time.sleep(3)  # rate limit courtesy
    if download_via_ytdlp(youtube_url, output_path):
        return True, "ytdlp"

    return False, "failed"


def download_mix_audio(mix, data_root, apify_token, use_apify=True):
    """Download mix audio. Returns (success, method)."""
    output_path = str(Path(data_root) / "mixes" / f"{mix['id']}.mp3")

    if os.path.isfile(output_path):
        return True, "cached"

    audio_url = mix.get("audio_url", "")
    if not audio_url:
        log.warning(f"No audio URL for mix {mix['id']}")
        return False, "no_url"

    # Try Apify first for YouTube URLs
    if use_apify and apify_token and "youtube.com" in audio_url:
        if download_via_apify(audio_url, output_path, apify_token):
            return True, "apify"

    # Fallback to yt-dlp
    from aidj.data.downloader import download_audio
    if download_audio(audio_url, output_path):
        return True, "ytdlp"

    return False, "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--data-root", type=str, default=str(config.DATA_ROOT))
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of mixes to download")
    parser.add_argument("--no-apify", action="store_true",
                        help="Skip Apify, use yt-dlp only")
    parser.add_argument("--apify-budget", type=float, default=5.0,
                        help="Max Apify spend in USD (approx, $0.015/track)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    apify_token = os.environ.get("APIFY_TOKEN", "")
    use_apify = bool(apify_token) and not args.no_apify

    if use_apify:
        max_apify_tracks = int(args.apify_budget / 0.015)
        log.info(f"Apify enabled: budget ${args.apify_budget:.2f} (~{max_apify_tracks} tracks)")
    else:
        max_apify_tracks = 0
        log.info("Apify disabled, using yt-dlp only")

    with open(args.manifest) as f:
        manifest = json.load(f)

    mixes = manifest["mixes"]
    if args.limit:
        mixes = mixes[:args.limit]

    data_root = Path(args.data_root)
    progress_path = data_root / "download_progress.json"
    progress = load_progress(progress_path)

    apify_count = 0
    stats = {"total": 0, "success": 0, "apify": 0, "ytdlp": 0, "cached": 0, "failed": 0}

    for mix in mixes:
        mix_id = mix["id"]

        # Download mix audio
        can_apify = use_apify and apify_count < max_apify_tracks
        ok, method = download_mix_audio(mix, str(data_root), apify_token, use_apify=can_apify)
        if method == "apify":
            apify_count += 1

        # Download tracks
        for track in mix.get("tracklist", []):
            tid = track.get("id")
            if tid is None:
                continue

            # Skip already completed
            if tid in progress["completed"]:
                stats["total"] += 1
                stats["success"] += 1
                stats["cached"] += 1
                continue

            stats["total"] += 1
            can_apify = use_apify and apify_count < max_apify_tracks
            ok, method = download_track(tid, str(data_root), apify_token, use_apify=can_apify)

            if ok:
                stats["success"] += 1
                stats[method] += 1
                progress["completed"][tid] = method
                if method == "apify":
                    apify_count += 1
            else:
                stats["failed"] += 1
                progress["failed"][tid] = True

            save_progress(progress, progress_path)

    print(f"\nResults:")
    print(f"  Total tracks: {stats['total']}")
    print(f"  Success:      {stats['success']} "
          f"(apify={stats['apify']}, ytdlp={stats['ytdlp']}, cached={stats['cached']})")
    print(f"  Failed:       {stats['failed']}")
    if use_apify:
        print(f"  Apify usage:  ~{apify_count} calls (~${apify_count * 0.015:.2f})")


if __name__ == "__main__":
    main()
