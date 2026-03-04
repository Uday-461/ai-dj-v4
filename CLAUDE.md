# AI DJ v4 -- Clean HuggingFace-First Pipeline

## Quick Start

```bash
cd v4
pip install -e ".[train,dev]"    # or: pip install -r requirements.txt
export AIDJ_DATA_ROOT=~/djmix    # data directory (default)
export HF_TOKEN=...              # HuggingFace token
export APIFY_TOKEN=...           # Apify token (optional, for downloads)
```

## Architecture

- **Data storage:** HuggingFace Hub (`Uday-4/djmix-v3`)
- **Heavy CPU work:** hp-mint (Ryzen 5 5600U, 12 threads, 14GB RAM, 75GB free)
- **GPU training:** Kaggle P100
- **Downloads:** Apify YouTube actor (primary, $0.015/track) + yt-dlp fallback

## Project Structure

```
v4/
  aidj/                      # library package
    config.py                # constants, paths, hyperparams
    analyzer.py              # BPM/key/energy analysis
    camelot.py               # Camelot key compatibility
    selector.py              # playlist ordering
    preprocessor.py          # cue detection, BPM sync
    assembler.py             # full mix assembly
    mixer.py                 # per-stem curve application + summing
    transition.py            # TransitionGenerator (stem-aware inference)
    data/                    # data pipeline modules
    stems/                   # Demucs separation + cache
    curves/                  # EQ filters + CVXPY optimizer
    model/                   # StemTransitionNet architecture + training
  scripts/
    01_select_subset.py      # select mixes from dataset JSON
    split_manifest.py        # split manifest into N parts for parallel processing
    02_download_audio.py     # Apify primary + yt-dlp fallback
    03_detect_beats.py       # beat detection
    04_align_tracks.py       # DTW alignment
    05_extract_transitions.py
    06_separate_stems.py     # Demucs stem separation (CPU on hp-mint)
    06b_compute_residuals.py
    07_extract_stem_curves.py # CVXPY optimization (CPU)
    08_build_dataset.py      # build train/val/test splits
    09_train.py              # model training (GPU)
    10_evaluate.py           # evaluation metrics
    11_generate_mix.py       # end-to-end inference
  hp_phase_a.py              # Phase A: download/beats/align/transitions + upload (hp-mint)
  hp_phase_b.py              # Phase B: stem curve extraction (hp-mint, legacy fallback)
  kaggle_phase_b.ipynb       # Phase B: Demucs + residuals, single notebook (legacy fallback)
  kaggle_phaseB_step6.ipynb  # Phase B Step 6: Demucs + residuals (Kaggle GPU, parallel-safe)
  kaggle_phaseB_step7.ipynb  # Phase B Step 7: CVXPY curves (Kaggle CPU, parallel-safe)
  kaggle_phase_c.ipynb       # Phase C: training + evaluation (Kaggle P100)
  upload_to_hf.py            # bulk upload to HF Hub
  setup_hp_mint.sh           # one-time hp-mint environment setup
  data/
    djmix-dataset.json       # full dataset (5,040 mixes)
    manifest_1mix.json       # 1 mix (dev)
    manifest_2mix.json       # 2 mixes (dev)
    manifest_50mix.json      # 50 mixes (original small run)
    manifest_100mix.json     # 100 mixes (production)
    manifest_100mix_part1.json # 50 mixes for hp-mint (odd indices)
    manifest_100mix_part2.json # 50 mixes for ops thinkpad (even indices)
```

## Pipeline Overview (Split Architecture)

The pipeline is split across different machines:

| Phase | Scripts | Where | Description |
|-------|---------|-------|-------------|
| **A. Data prep** | 02-05 | hp-mint | Download, beats, align, transitions -> upload to HF |
| **B. Stems** | 06, 06b, 06c | Kaggle P100 | Demucs separation, residuals (`kaggle_phaseB_step6.ipynb` x2) |
| **B. Curves** | 07 | Kaggle CPU | CVXPY curve extraction (`kaggle_phaseB_step7.ipynb` x2) |
| **C. Train** | 08-10 | Kaggle P100 | Build dataset, train StemTransitionNet, evaluate (`kaggle_phase_c.ipynb`) |
| D. Infer | 11 | any | Generate transition for new track pair |

### Phase A: `hp_phase_a.py` (hp-mint)

Processes 1 mix at a time on CPU, uploads to HF Hub, cleans up local disk.

```bash
# Start Phase A (50 mixes)
nohup python3 hp_phase_a.py --manifest data/manifest_50mix.json > ~/phase_a.log 2>&1 &

# Resume after interruption (auto-skips completed mixes)
python3 hp_phase_a.py --manifest data/manifest_50mix.json

# Skip to a specific mix
python3 hp_phase_a.py --manifest data/manifest.json --skip-to mix4527

# Use yt-dlp only (no Apify cost)
python3 hp_phase_a.py --manifest data/manifest.json --no-apify

# Run a subset manifest (e.g. retries + N new mixes)
python3 hp_phase_a.py --manifest data/manifest_retry10.json

# Parallel Phase A on two machines (100 mixes, 50 each):
# hp-mint:
nohup python3 hp_phase_a.py --manifest data/manifest_100mix_part1.json --no-apify > ~/phase_a.log 2>&1 &
# ops thinkpad:
nohup python3 hp_phase_a.py --manifest data/manifest_100mix_part2.json --no-apify > ~/phase_a.log 2>&1 &
```

Features:
- Resumable via `DATA_ROOT/phase_a_progress.json`
- Uses `upload_large_folder` (batches 50 files/commit, handles 429s internally)
- yt-dlp downloads at 128kbps mono (~4MB/track, halved vs default)
- Deletes local files after each mix (respects shared tracks across mixes)
- Loads tokens from `.env` (HF_TOKEN, APIFY_TOKEN)

Uploads to HF per mix:
- `tracks/{id}.mp3` -- track audio (128kbps mono)
- `mixes/{mix_id}.mp3` -- mix audio
- `beats/{id}.npz` -- beat timestamps
- `results/alignments/{mix_id}.pkl` -- DTW alignments
- `results/transitions/{mix_id}.pkl` -- transition regions

**HF rate limit note:** HF allows 256 commits/hour. `upload_large_folder` handles
this automatically by retrying with smaller batches. If you see `Failed to commit`
in the log, it's expected -- the library is backing off and will succeed once the
window resets (~1 hour max wait).

**hp-mint suspend fix:** Run once to prevent the machine from suspending:
```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

**Manifests:**
- `data/manifest_1mix.json` -- 1 house mix (Jimpster, mix0502)
- `data/manifest_2mix.json` -- 2 mixes (dev)
- `data/manifest_50mix.json` -- 50 mixes (original small run)
- `data/manifest_100mix.json` -- 100 mixes, 3928 tracks, 2931 transitions (production)
- `data/manifest_100mix_part1.json` -- 50 mixes for hp-mint (round-robin odd)
- `data/manifest_100mix_part2.json` -- 50 mixes for ops thinkpad (round-robin even)

**Monitoring:**
```bash
# Watch for completions and errors
grep -E '(Uploaded|Upload failed|ERROR|INFO \[)' ~/phase_a.log | tail -20
# Check progress
python3 -c "import json; p=json.load(open('djmix/phase_a_progress.json')); print('Done:', len(p['completed_mixes'])); print('Failed:', list(p['failed'].keys())); print('Stats:', p['stats'])"
```

### Phase B: 2 Parallel Kaggle Notebooks

**Step 6 (GPU):** `kaggle_phaseB_step6.ipynb`
- Upload to Kaggle, set Accelerator = GPU P100
- Edit MANIFEST in Cell 2: `"manifest_100mix_part1"` or `"manifest_100mix_part2"`
- Run cells 1-5. Resumable across 12h sessions.
- Run 2 notebooks in parallel (one per manifest part)
- Each writes its own progress file (`phase_b_progress_step6_part1.json` / `part2`)

**Step 7 (CPU):** `kaggle_phaseB_step7.ipynb`
- Upload to Kaggle, set Accelerator = None (CPU)
- Edit MANIFEST in Cell 2: same as above
- Run after Step 6 completes for your manifest part
- Run 2 notebooks in parallel
- Each writes its own progress file (`phase_b_progress_step7_part1.json` / `part2`)

**DRY_RUN:** Both notebooks have `DRY_RUN = False` in Cell 2. Set `True` to process only 1 mix (quick error check before full run).

**Progress isolation:** Each notebook writes only its own part-scoped progress file. To check global completion, each notebook merges all progress files from HF. No race conditions — each file has exactly one writer.

**Legacy fallbacks (kept, not deleted):**
- `kaggle_phase_b.ipynb` — original single-notebook approach
- `hp_phase_b.py` — curve extraction on hp-mint CPU

```bash
# Legacy: run curves on hp-mint instead of Kaggle
nohup python3 hp_phase_b.py --manifest data/manifest_50mix.json > ~/phase_b.log 2>&1 &
```

### Phase C: Training (Kaggle P100)

**Kaggle notebook (`kaggle_phase_c.ipynb`):**
- Clones repo, downloads dataset from HF
- Runs 08_build_dataset, 09_train, 10_evaluate
- Uploads trained model to HF Hub (`Uday-4/aidj-v3-stemtransitionnet`)

## Cost & Time Estimates (500 mixes, ~7500 tracks, ~3500 transitions)

### Phase A: Data prep (steps 02-05) -- hp-mint CPU

| Step | Time/unit | Bottleneck | Total est. | Cost |
|------|-----------|------------|------------|------|
| 02 download | ~10s/track (Apify), ~30-40s (yt-dlp) | Network I/O + API latency | 21-83h | Apify: ~$112 (7500 x $0.015); yt-dlp: $0 |
| 03 beats | ~8s/track | librosa CPU (BeatNet unavailable on hp-mint) | ~17h | $0 |
| 04 align | ~3s/pair | DTW on mel spectrograms | ~6h | $0 |
| 05 transitions | ~1s/mix | Simple array slicing | <1h | $0 |

**Phase A total: ~45-107h (~2-4 days), $0-112** (yt-dlp is free but slower)

### Phase B: Stems + Curves -- Kaggle P100 + hp-mint

| Step | Time/unit (GPU) | Time/unit (CPU) | Bottleneck | Total (GPU) | Cost (GPU) |
|------|-----------------|-----------------|------------|-------------|------------|
| 06 Demucs tracks | 30s/track | 5 min/track | Model inference; 2GB RAM per track; GPU gives 10x speedup | ~63h | free (Kaggle) |
| 06 Demucs mix segs | 6s/segment | 1 min/segment | Short audio segments (30-60s) | ~6h | included |
| 06b residuals | ~3s/transition | same | I/O bound, mix_stems - track_stems | ~3h | included |
| 07 curves (ECOS) | ~45s/transition | same | CVXPY convex opt; 8 vars x num_frames per stem | ~44h | $0 (hp-mint) |

**Phase B total: ~70-116h (~3-5 days), $0** (Kaggle free tier + hp-mint CPU)

**Optimizations applied:**
- `OPT_HOP` increased 4x (4096 -> 16384), reducing variables per transition by 4x
- Transitions capped at `MAX_TRANSITION_SECS = 60s` before optimization
- Step 07 supports `--workers N` for multiprocessing (default: CPU_count / 2)

### Phase C: Train (steps 08-10) -- Kaggle P100

| Step | Time/unit | Total est. | Cost |
|------|-----------|------------|------|
| 08 build dataset | ~5 min | 5 min | free |
| 09 train (50 epochs) | ~1-2h | 1-2h | free |
| 10 evaluate | ~5 min | 5 min | free |

**Phase C total: ~1-2h, free** (Kaggle free tier, 30h/week GPU quota)

### Total pipeline estimate (split architecture)

| Where | What | Time | Cost |
|-------|------|------|------|
| hp-mint | Phase A (download, beats, align, transitions) | ~2-4 days | $0-112 (Apify) |
| Kaggle P100 + hp-mint | Phase B (Demucs, residuals, curves) | ~3-5 days | $0 |
| Kaggle P100 | Phase C (train, evaluate) | ~2h | $0 |
| **Total** | | **~5-9 days** (parallelized) | **$0-112** |

Key cost driver: Apify ($0.015/track) -- use yt-dlp to reduce to $0 at cost of 3x slower downloads.
Key time drivers: Demucs GPU separation (~55% of Phase B), CVXPY solver (~40% if ECOS works).

## hp-mint Workflow

```bash
# One-time setup
bash setup_hp_mint.sh

# Phase A: data prep
source ~/aidj-venv/bin/activate
cd ~/ai-dj/v4
python hp_phase_a.py --manifest data/manifest_50mix.json

# Phase B curves (after Kaggle stems are uploaded)
python hp_phase_b.py --manifest data/manifest_50mix.json
```

## Downloads

Script `02_download_audio.py` uses Apify as primary downloader:
- Apify actor: `marielise.dev~youtube-video-downloader`
- Cost: ~$0.015/track, configurable budget cap (default $5)
- Automatic yt-dlp fallback when Apify fails or budget exhausted
- Progress tracked in `DATA_ROOT/download_progress.json`

## Subset Selection (`aidj/data/subset_selector.py`)

`01_select_subset.py` uses `select_subset()` to filter and rank mixes from the full dataset:
- Filters: `audio_source` must be soundcloud/youtube, `num_available_transitions >= 3`, track coverage >= 50%
- Sorted by `num_available_transitions` descending, takes top N (default `SUBSET_SIZE = 500`)
- Adds `genres` field from tags
- Output: manifest JSON with `{num_mixes, num_tracks, num_transitions, mixes}`

Pre-built manifests:
- `data/manifest_2mix.json` -- 2 mixes (dev)
- `data/manifest_1mix.json` -- 1 house mix (Jimpster, mix0502)
- `data/manifest_50mix.json` -- 50 house mixes, 1321 tracks, 1055 transitions

## Key Configuration (`aidj/config.py`)

- `AIDJ_DATA_ROOT` env var controls data directory (default `~/djmix`)
- `SR = 44100`, `FEATURE_SR = 22050` (features use half rate to save RAM)
- `STEMS = ["drums", "bass", "vocals", "other"]`
- `STEM_FORMAT = "ogg"` -- stems stored as OGG on HuggingFace
- `DEMUCS_MODEL = "htdemucs"`
- `OPT_HOP = 16384`, `OPT_FFT = 16384` -- CVXPY optimizer hop (4x from original 4096)
- `MAX_TRANSITION_SECS = 60` -- cap for curve extraction

## Conventions

- Scripts numbered 01-11, run sequentially
- Each script accepts `--manifest` and `--data-root` flags
- All audio at 44.1kHz mono
- No pandas dependency
- BeatNet for beat detection, librosa fallback
