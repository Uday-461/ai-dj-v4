#!/usr/bin/env python3
"""Phase A pipeline for hp-mint: Download → Beats → Align → Transitions → Upload to HF.

Processes one mix at a time to manage disk space. After each mix is uploaded
to HuggingFace Hub, local files are deleted.

Phase B (RunPod GPU) picks up from HF and runs Demucs → Residuals → Curves.

Usage:
    python hp_phase_a.py --manifest data/manifest.json
    python hp_phase_a.py --manifest data/manifest.json --skip-to mix4527
    python hp_phase_a.py --manifest data/manifest.json --no-apify
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aidj import config

log = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent / "scripts"
PROGRESS_FILE = "phase_a_progress.json"


def load_progress(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "completed_mixes": [],
        "failed": {},
        "stats": {
            "mixes_done": 0,
            "tracks_downloaded": 0,
            "tracks_failed": 0,
            "transitions_extracted": 0,
            "total_time_hrs": 0,
        },
    }


def save_progress(progress, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def run_script(script_name, args_list):
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + args_list
    log.info(f"Running: {script_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"STDOUT: {result.stdout[-500:]}" if result.stdout else "")
        log.error(f"STDERR: {result.stderr[-500:]}" if result.stderr else "")
        raise RuntimeError(f"{script_name} failed with code {result.returncode}")
    return result.stdout


def create_single_mix_manifest(mix, path):
    with open(path, "w") as f:
        json.dump({"mixes": [mix]}, f, indent=2)


def get_track_ids(mix):
    return [t["id"] for t in mix.get("tracklist", []) if t.get("id")]


def upload_mix_to_hf(mix, data_root, hf_repo, hf_token):
    """Upload all Phase A outputs for one mix to HF Hub."""
    import shutil
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_repo, repo_type="dataset", private=True, exist_ok=True)

    mix_id = mix["id"]

    # Stage all files into a temp dir mirroring the repo layout,
    # then upload_folder to produce exactly one commit per mix.
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        uploaded = 0

        def stage(src, rel_path):
            nonlocal uploaded
            if src.exists():
                dst = staging / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                uploaded += 1

        # Mix audio
        stage(data_root / "mixes" / f"{mix_id}.mp3", f"mixes/{mix_id}.mp3")

        # Track audio + beats
        for tid in get_track_ids(mix):
            stage(data_root / "tracks" / f"{tid}.mp3", f"tracks/{tid}.mp3")
            stage(data_root / "beats" / f"{tid}.npz", f"beats/{tid}.npz")

        # Alignment + transitions
        stage(data_root / "results" / "alignments" / f"{mix_id}.pkl",
              f"results/alignments/{mix_id}.pkl")
        stage(data_root / "results" / "transitions" / f"{mix_id}.pkl",
              f"results/transitions/{mix_id}.pkl")

        if uploaded == 0:
            log.warning(f"No files to upload for mix {mix_id}")
            return 0

        api.upload_large_folder(
            folder_path=str(staging),
            repo_id=hf_repo,
            repo_type="dataset",
        )

    log.info(f"Uploaded {uploaded} files for mix {mix_id} (1 commit)")
    return uploaded


def cleanup_mix_files(mix, data_root, all_mixes, completed_mixes):
    """Delete local files for a mix to free disk space.

    Only deletes track files if the track isn't needed by a pending mix.
    """
    mix_id = mix["id"]
    deleted = 0

    # Delete mix audio
    mix_path = data_root / "mixes" / f"{mix_id}.mp3"
    if mix_path.exists():
        mix_path.unlink()
        deleted += 1

    # Find tracks used by other pending mixes
    pending_track_ids = set()
    completed_set = set(completed_mixes)
    for m in all_mixes:
        if m["id"] != mix_id and m["id"] not in completed_set:
            for t in m.get("tracklist", []):
                if t.get("id"):
                    pending_track_ids.add(t["id"])

    # Delete tracks not needed by pending mixes
    for tid in get_track_ids(mix):
        if tid not in pending_track_ids:
            track_path = data_root / "tracks" / f"{tid}.mp3"
            if track_path.exists():
                track_path.unlink()
                deleted += 1

    # Delete beat files (already uploaded)
    for tid in get_track_ids(mix):
        if tid not in pending_track_ids:
            beat_path = data_root / "beats" / f"{tid}.npz"
            if beat_path.exists():
                beat_path.unlink()
                deleted += 1

    mix_beat = data_root / "beats" / f"{mix_id}.npz"
    if mix_beat.exists():
        mix_beat.unlink()
        deleted += 1

    # Delete alignment and transition results (already uploaded)
    for subdir in ["alignments", "transitions"]:
        p = data_root / "results" / subdir / f"{mix_id}.pkl"
        if p.exists():
            p.unlink()
            deleted += 1

    log.info(f"Cleaned up {deleted} files for mix {mix_id}")


def process_mix(mix, data_root, tmp_manifest, all_mixes, hf_repo, hf_token,
                progress, progress_path):
    """Process one mix: download → beats → align → transitions → upload → cleanup."""
    mix_id = mix["id"]
    start = time.time()

    create_single_mix_manifest(mix, tmp_manifest)
    common_args = ["--manifest", str(tmp_manifest), "--data-root", str(data_root)]

    steps = [
        ("02_download_audio.py", []),
        ("03_detect_beats.py", ["--skip-mix-audio"]),
        ("04_align_tracks.py", []),
        ("05_extract_transitions.py", []),
    ]

    for script, extra_args in steps:
        try:
            run_script(script, common_args + extra_args)
        except RuntimeError as e:
            log.error(f"{script} failed for {mix_id}: {e}")
            progress["failed"][mix_id] = f"{script}: {e}"
            save_progress(progress, progress_path)
            return False

    # Count results
    tran_path = data_root / "results" / "transitions" / f"{mix_id}.pkl"
    n_transitions = 0
    if tran_path.exists():
        import pickle
        with open(tran_path, "rb") as f:
            n_transitions = len(pickle.load(f))

    n_tracks = len([1 for tid in get_track_ids(mix)
                    if (data_root / "tracks" / f"{tid}.mp3").exists()])

    # Upload to HF
    if hf_token:
        try:
            upload_mix_to_hf(mix, data_root, hf_repo, hf_token)
        except Exception as e:
            log.error(f"Upload failed for {mix_id}: {e}")
            progress["failed"][mix_id] = f"upload: {e}"
            save_progress(progress, progress_path)
            return False

    # Cleanup local files
    cleanup_mix_files(mix, data_root, all_mixes, progress["completed_mixes"])

    elapsed = (time.time() - start) / 3600
    progress["completed_mixes"].append(mix_id)
    progress["stats"]["mixes_done"] += 1
    progress["stats"]["tracks_downloaded"] += n_tracks
    progress["stats"]["transitions_extracted"] += n_transitions
    progress["stats"]["total_time_hrs"] = round(
        progress["stats"]["total_time_hrs"] + elapsed, 2)
    save_progress(progress, progress_path)

    log.info(f"Mix {mix_id}: {n_tracks} tracks, {n_transitions} transitions, "
             f"{elapsed:.2f}h")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    parser.add_argument("--data-root", type=str, default=str(config.DATA_ROOT))
    parser.add_argument("--hf-repo", type=str, default="Uday-4/djmix-v3")
    parser.add_argument("--skip-to", type=str, default=None,
                        help="Skip mixes until this mix ID")
    parser.add_argument("--no-apify", action="store_true",
                        help="Disable Apify, use yt-dlp only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    os.environ["AIDJ_DATA_ROOT"] = str(data_root)

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        log.warning("HF_TOKEN not set — will process but not upload")

    with open(args.manifest) as f:
        manifest = json.load(f)

    all_mixes = manifest["mixes"]
    progress_path = data_root / PROGRESS_FILE
    progress = load_progress(progress_path)

    # Filter to pending mixes
    completed = set(progress["completed_mixes"])
    pending = [m for m in all_mixes if m["id"] not in completed]

    if args.skip_to:
        skip_ids = set()
        for m in pending:
            if m["id"] == args.skip_to:
                break
            skip_ids.add(m["id"])
        pending = [m for m in pending if m["id"] not in skip_ids]

    log.info(f"Phase A: {len(all_mixes)} total, {len(completed)} done, "
             f"{len(pending)} pending")

    tmp_manifest = data_root / "_tmp_phase_a_manifest.json"

    for i, mix in enumerate(pending):
        log.info(f"\n{'='*60}")
        log.info(f"[{i+1}/{len(pending)}] Mix {mix['id']} "
                 f"({len(get_track_ids(mix))} tracks)")
        log.info(f"{'='*60}")

        process_mix(mix, data_root, tmp_manifest, all_mixes,
                    args.hf_repo, hf_token, progress, progress_path)

    if tmp_manifest.exists():
        tmp_manifest.unlink()

    # Retry failed mixes once (catches upload rate-limit failures and transient errors)
    if progress["failed"]:
        failed_ids = list(progress["failed"].keys())
        log.info(f"\nRetrying {len(failed_ids)} failed mixes: {failed_ids}")
        mix_by_id = {m["id"]: m for m in all_mixes}
        for mix_id in failed_ids:
            mix = mix_by_id.get(mix_id)
            if mix is None:
                continue
            log.info(f"\nRetry: {mix_id}")
            del progress["failed"][mix_id]
            save_progress(progress, progress_path)
            process_mix(mix, data_root, tmp_manifest, all_mixes,
                        args.hf_repo, hf_token, progress, progress_path)
        if tmp_manifest.exists():
            tmp_manifest.unlink()

    s = progress["stats"]
    print(f"\nPhase A complete!")
    print(f"  Mixes:       {s['mixes_done']}")
    print(f"  Tracks:      {s['tracks_downloaded']}")
    print(f"  Transitions: {s['transitions_extracted']}")
    print(f"  Time:        {s['total_time_hrs']:.1f}h")
    if progress["failed"]:
        print(f"  Failed:      {len(progress['failed'])}")
        for mid, reason in progress["failed"].items():
            print(f"    {mid}: {reason}")


if __name__ == "__main__":
    main()
