#!/usr/bin/env python3
"""Phase B step 07: Download stems from HF → Extract stem curves → Upload curves to HF.

Runs on hp-mint (CPU). For each mix:
  1. Download track stems + mix segment stems from HF
  2. Run CVXPY optimizer per transition (multiprocessing)
  3. Upload curves to HF
  4. Delete local stems to free disk space

Progress tracked in phase_b_progress.json on HF (same file Kaggle uses).
Only the "curves" key is written by this script.

Usage:
    python hp_phase_b.py --manifest data/manifest_50mix.json
    python hp_phase_b.py --manifest data/manifest_50mix.json --workers 6
    python hp_phase_b.py --manifest data/manifest_50mix.json --skip-to mix1195
"""
import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

# Suppress HF download progress bars so the log stays clean (no carriage-return spam)
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aidj import config
from aidj.curves.stem_curve_extractor import StemCurveExtractor
from aidj.stems.stem_cache import StemCache

log = logging.getLogger(__name__)

HF_REPO = "Uday-4/djmix-v3"
PROGRESS_KEY = "phase_b_progress.json"


# -------------------------------------------------------
# Progress helpers (shared with Kaggle notebook)
# -------------------------------------------------------

def load_progress(data_root, hf_token):
    from huggingface_hub import hf_hub_download
    local_path = data_root / PROGRESS_KEY
    try:
        hf_hub_download(
            repo_id=HF_REPO, filename=PROGRESS_KEY,
            repo_type="dataset", token=hf_token,
            local_dir=str(data_root), force_download=True,
        )
        with open(local_path) as f:
            p = json.load(f)
        log.info(f"Loaded progress from HF: {PROGRESS_KEY}")
        return p
    except Exception as e:
        log.warning(f"No progress file on HF ({e}), starting fresh")
        return {"stems_tracks": [], "stems_segments": [], "residuals": [], "curves": []}


def push_progress(progress, data_root, hf_token):
    from huggingface_hub import HfApi
    local_path = data_root / PROGRESS_KEY
    with open(local_path, "w") as f:
        json.dump(progress, f, indent=2)
    HfApi(token=hf_token).upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=PROGRESS_KEY,
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message="Phase B step 07 progress",
    )


# -------------------------------------------------------
# HF download helpers
# -------------------------------------------------------

def download_track_stems(tid, data_root, hf_token):
    """Download all 4 stem OGGs for a track from HF. Returns True if all present."""
    from huggingface_hub import hf_hub_download
    stem_dir = data_root / "stems" / "tracks" / tid
    ext = config.STEM_EXT[config.STEM_FORMAT]
    all_ok = True
    for stem in config.STEMS:
        local = stem_dir / f"{stem}{ext}"
        if local.exists():
            continue
        stem_dir.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=HF_REPO,
                filename=f"stems/tracks/{tid}/{stem}{ext}",
                repo_type="dataset", token=hf_token,
                local_dir=str(data_root),
            )
        except Exception as e:
            log.warning(f"  Could not download stem {stem} for track {tid}: {e}")
            all_ok = False
    return all_ok


def download_mix_seg_stems(tran_id, data_root, hf_token):
    """Download all 4 stem OGGs for a mix segment from HF. Returns True if all present."""
    from huggingface_hub import hf_hub_download
    seg_dir = data_root / "stems" / "mix_segments" / tran_id
    ext = config.STEM_EXT[config.STEM_FORMAT]
    all_ok = True
    for stem in config.STEMS:
        local = seg_dir / f"{stem}{ext}"
        if local.exists():
            continue
        seg_dir.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=HF_REPO,
                filename=f"stems/mix_segments/{tran_id}/{stem}{ext}",
                repo_type="dataset", token=hf_token,
                local_dir=str(data_root),
            )
        except Exception as e:
            log.warning(f"  Could not download stem {stem} for segment {tran_id}: {e}")
            all_ok = False
    return all_ok


def download_transition_pkl(mix_id, data_root, hf_token):
    from huggingface_hub import hf_hub_download
    local = data_root / "results" / "transitions" / f"{mix_id}.pkl"
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=HF_REPO,
        filename=f"results/transitions/{mix_id}.pkl",
        repo_type="dataset", token=hf_token,
        local_dir=str(data_root),
    )
    return local


# -------------------------------------------------------
# Worker: extract curves for one transition (multiprocessing)
# -------------------------------------------------------

def _extract_one(args):
    tran, data_root_str, output_path_str = args
    tran_id = tran["tran_id"]
    prev_id = tran["track_id_prev"]
    next_id = tran["track_id_next"]

    data_root = Path(data_root_str)
    output_path = Path(output_path_str)

    if output_path.exists():
        return tran_id, True, "cached"

    # Sanity check cue times BEFORE loading any stems (avoids hanging on bogus transitions)
    prev_cue_secs = tran.get("track_cue_in_time_prev", 0)
    next_cue_secs = tran.get("track_cue_in_time_next", 0)
    MAX_CUE_SECS = 3600  # tracks are never longer than 1 hour
    if prev_cue_secs > MAX_CUE_SECS or next_cue_secs > MAX_CUE_SECS:
        return tran_id, False, (
            f"bogus cue times: prev_cue={prev_cue_secs:.0f}s, "
            f"next_cue={next_cue_secs:.0f}s (max={MAX_CUE_SECS}s)"
        )

    stem_cache = StemCache(data_root)

    mix_seg_stems = stem_cache.load_stems("mix_segments", tran_id)
    prev_stems    = stem_cache.load_stems("tracks", prev_id)
    next_stems    = stem_cache.load_stems("tracks", next_id)

    if mix_seg_stems is None or prev_stems is None or next_stems is None:
        missing = []
        if mix_seg_stems is None: missing.append(f"mix_seg/{tran_id}")
        if prev_stems is None:    missing.append(f"track/{prev_id}")
        if next_stems is None:    missing.append(f"track/{next_id}")
        return tran_id, False, f"missing stems: {', '.join(missing)}"

    try:
        region_len = min(len(v) for v in mix_seg_stems.values())
        max_samples = int(config.MAX_TRANSITION_SECS * config.OPT_SR)
        region_len = min(region_len, max_samples)

        if region_len < config.OPT_SR:
            return tran_id, False, "too short"

        mix_region = {s: mix_seg_stems[s][:region_len] for s in config.STEMS}

        prev_start = int(prev_cue_secs * config.OPT_SR)
        next_start = int(next_cue_secs * config.OPT_SR)

        # Secondary check: cue start must be within the loaded track audio
        prev_track_len = min(len(v) for v in prev_stems.values())
        next_track_len = min(len(v) for v in next_stems.values())
        if prev_start >= prev_track_len or next_start >= next_track_len:
            return tran_id, False, (
                f"cue out of range: prev_start={prev_start/config.OPT_SR:.0f}s "
                f"(track={prev_track_len/config.OPT_SR:.0f}s), "
                f"next_start={next_start/config.OPT_SR:.0f}s "
                f"(track={next_track_len/config.OPT_SR:.0f}s)"
            )

        prev_region = {s: prev_stems[s][prev_start:prev_start + region_len] for s in config.STEMS}
        next_region = {s: next_stems[s][next_start:next_start + region_len] for s in config.STEMS}

        min_len = min(
            region_len,
            min(len(v) for v in prev_region.values()),
            min(len(v) for v in next_region.values()),
        )
        if min_len < config.OPT_SR:
            return tran_id, False, "too short after trim"

        mix_region  = {s: v[:min_len] for s, v in mix_region.items()}
        prev_region = {s: v[:min_len] for s, v in prev_region.items()}
        next_region = {s: v[:min_len] for s, v in next_region.items()}

        extractor = StemCurveExtractor()
        curves = extractor.extract_transition_curves(mix_region, prev_region, next_region)

        if curves is not None:
            extractor.save_curves(curves, output_path)
            return tran_id, True, None
        else:
            return tran_id, False, "optimization returned None"

    except Exception as e:
        return tran_id, False, str(e)


# -------------------------------------------------------
# Per-mix processing
# -------------------------------------------------------

def process_mix(mix_id, transitions, data_root, hf_token, n_workers, curves_dir):
    """Download stems, extract curves, return (n_ok, n_total)."""
    from huggingface_hub import list_repo_files

    # Collect unique track IDs needed for this mix
    track_ids = set()
    for t in transitions:
        if t.get("track_id_prev"): track_ids.add(t["track_id_prev"])
        if t.get("track_id_next"): track_ids.add(t["track_id_next"])

    # --- Download stems in parallel (I/O bound, threads are fine) ---
    from concurrent.futures import ThreadPoolExecutor
    log.info(f"  Downloading stems for {len(track_ids)} tracks + {len(transitions)} segments...")
    with ThreadPoolExecutor(max_workers=8) as tp:
        tp.map(lambda tid: download_track_stems(tid, data_root, hf_token), sorted(track_ids))
        tp.map(lambda t: download_mix_seg_stems(t["tran_id"], data_root, hf_token), transitions)

    # --- Build work items ---
    mix_curves_dir = curves_dir / mix_id
    mix_curves_dir.mkdir(parents=True, exist_ok=True)

    work_items = []
    already_done = 0
    for tran in transitions:
        tran_id = tran["tran_id"]
        out = mix_curves_dir / f"{tran_id}.npz"
        if out.exists():
            already_done += 1
        else:
            work_items.append((tran, str(data_root), str(out)))

    log.info(f"  Curve extraction: {len(work_items)} pending, {already_done} cached")

    # --- Run optimizer ---
    n_ok = already_done
    n_fail = 0
    total = len(transitions)

    WORKER_TIMEOUT = 90  # seconds — ECOS normally finishes in <15s; kill anything longer

    if work_items:
        # Submit all items in parallel, then collect with per-item timeout.
        # If any worker hangs in a C extension, terminate the whole pool and
        # restart with a fresh pool for the remaining items.
        remaining = list(enumerate(work_items))
        while remaining:
            with Pool(min(n_workers, len(remaining))) as pool:
                futures = []
                for idx, item in remaining:
                    futures.append((idx, item[0]["tran_id"], pool.apply_async(_extract_one, (item,))))

                new_remaining = []
                pool_killed = False
                for idx, tran_id, future in futures:
                    if pool_killed:
                        new_remaining.append((idx, work_items[idx]))
                        continue
                    try:
                        _, ok, err = future.get(timeout=WORKER_TIMEOUT)
                        if ok:
                            n_ok += 1
                            log.info(f"  [{idx+1}/{len(work_items)}] {tran_id}: OK")
                        else:
                            n_fail += 1
                            log.warning(f"  [{idx+1}/{len(work_items)}] {tran_id}: {err}")
                    except Exception:
                        n_fail += 1
                        log.warning(f"  [{idx+1}/{len(work_items)}] {tran_id}: timeout ({WORKER_TIMEOUT}s), skipping")
                        pool.terminate()
                        pool.join()
                        pool_killed = True
                remaining = new_remaining

    log.info(f"  Curves: {n_ok} ok, {n_fail} failed, {total} total")
    return n_ok, total


def upload_curves(mix_id, curves_dir, hf_token):
    """Upload curve npz files for one mix to HF."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    mix_curves_dir = curves_dir / mix_id
    curve_files = list(mix_curves_dir.glob("*.npz"))
    if not curve_files:
        log.warning(f"  No curve files to upload for {mix_id}")
        return 0

    log.info(f"  Uploading {len(curve_files)} curves for {mix_id}...")
    for f in curve_files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"results/stem_curves/{mix_id}/{f.name}",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
    log.info(f"  Upload done ({len(curve_files)} files)")
    return len(curve_files)


def cleanup_mix_stems(mix_id, transitions, data_root, all_pending_track_ids):
    """Delete local stem files for this mix to free disk.

    Keeps track stems that are still needed by later mixes.
    """
    deleted = 0

    # Mix segment stems — always delete after curves extracted
    for tran in transitions:
        seg_dir = data_root / "stems" / "mix_segments" / tran["tran_id"]
        if seg_dir.exists():
            shutil.rmtree(seg_dir)
            deleted += 1

    # Track stems — only delete if not needed by remaining mixes
    track_ids = set()
    for t in transitions:
        if t.get("track_id_prev"): track_ids.add(t["track_id_prev"])
        if t.get("track_id_next"): track_ids.add(t["track_id_next"])

    for tid in track_ids:
        if tid not in all_pending_track_ids:
            stem_dir = data_root / "stems" / "tracks" / tid
            if stem_dir.exists():
                shutil.rmtree(stem_dir)
                deleted += 1

    log.info(f"  Cleaned up {deleted} stem dirs for {mix_id}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", default="data/manifest_50mix.json")
    parser.add_argument("--data-root", default=str(config.DATA_ROOT))
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel CVXPY workers (default: CPU count / 2)")
    parser.add_argument("--skip-to", default=None,
                        help="Skip mixes until this mix ID")
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
        log.error("HF_TOKEN not set — cannot download stems from HF")
        sys.exit(1)

    n_workers = args.workers or max(1, (os.cpu_count() or 2) // 2)
    log.info(f"Workers: {n_workers}")

    with open(args.manifest) as f:
        manifest = json.load(f)
    all_mixes = manifest["mixes"]

    # Load progress from HF
    progress = load_progress(data_root, hf_token)
    curves_done = set(progress["curves"])

    # Filter to mixes that have residuals done (phase B step 06c) but not curves
    stems_done = set(progress["stems_tracks"]) & set(progress["stems_segments"]) \
                 & set(progress["residuals"])
    pending = [m for m in all_mixes
               if m["id"] in stems_done and m["id"] not in curves_done]

    if args.skip_to:
        idx = next((i for i, m in enumerate(pending) if m["id"] == args.skip_to), None)
        if idx is not None:
            pending = pending[idx:]
        else:
            log.warning(f"--skip-to {args.skip_to} not found in pending list")

    log.info(f"Mixes ready for step 07: {len(pending)} pending, "
             f"{len(curves_done)} already done")

    if not pending:
        log.info("Nothing to do. Run Kaggle notebook first to complete step 06.")
        return

    curves_dir = data_root / "results" / "stem_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    # Download all transition PKLs upfront
    log.info("Downloading transition PKLs...")
    mix_transitions = {}
    for mix in all_mixes:
        mix_id = mix["id"]
        try:
            pkl = download_transition_pkl(mix_id, data_root, hf_token)
            with open(pkl, "rb") as f:
                mix_transitions[mix_id] = pickle.load(f)
        except Exception as e:
            log.warning(f"Could not download transitions for {mix_id}: {e}")

    session_start = time.time()

    for mix_idx, mix in enumerate(pending):
        mix_id = mix["id"]
        transitions = mix_transitions.get(mix_id)
        if not transitions:
            log.warning(f"[{mix_idx+1}/{len(pending)}] {mix_id}: no transitions, skipping")
            continue

        mix_start = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"[{mix_idx+1}/{len(pending)}] {mix_id} "
                 f"({len(transitions)} transitions)")
        log.info(f"{'='*60}")

        # Track IDs still needed by mixes after this one
        remaining_mixes = pending[mix_idx + 1:]
        remaining_track_ids = set()
        for m in remaining_mixes:
            m_tran = mix_transitions.get(m["id"], [])
            for t in m_tran:
                if t.get("track_id_prev"): remaining_track_ids.add(t["track_id_prev"])
                if t.get("track_id_next"): remaining_track_ids.add(t["track_id_next"])

        try:
            n_ok, n_total = process_mix(
                mix_id, transitions, data_root, hf_token, n_workers, curves_dir
            )
        except Exception as e:
            log.error(f"process_mix failed for {mix_id}: {e}")
            continue

        # Upload curves
        try:
            upload_curves(mix_id, curves_dir, hf_token)
        except Exception as e:
            log.error(f"Upload failed for {mix_id}: {e}")
            # Don't mark done — retry next run
            continue

        # Mark done only if ≥1 curve produced
        if n_ok > 0:
            progress["curves"].append(mix_id)
            push_progress(progress, data_root, hf_token)
        else:
            log.warning(f"  0 curves produced for {mix_id}, NOT marking done")

        # Cleanup local stems
        cleanup_mix_stems(mix_id, transitions, data_root, remaining_track_ids)

        elapsed = (time.time() - mix_start) / 60
        total_elapsed = (time.time() - session_start) / 3600
        log.info(f"[{mix_idx+1}/{len(pending)}] {mix_id} done in "
                 f"{elapsed:.1f} min | session: {total_elapsed:.2f}h")

    print(f"\nPhase B step 07 complete!")
    print(f"  Curves done: {len(progress['curves'])}/{len(all_mixes)} mixes")


if __name__ == "__main__":
    main()
