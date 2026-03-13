# Step 7: Resumable Stem Curve Extraction Guide

This guide explains how to use the new manifest-based, resumable approach for Step 7 curve extraction.

## Overview

**Goal:** Extract per-stem fader + EQ curves for ALL available mixes/residuals on HuggingFace, supporting:
- Parallel processing on 2 Kaggle notebooks (50/50 split)
- Resumable execution across 12h Kaggle session timeouts
- Comprehensive logging showing progress, status, and decisions
- Full coverage of partial data (not just complete 100-mix manifests)

**Key Architecture:**
1. **Create training manifest** → queries HF for ALL available residuals
2. **Split manifest** → creates part1 (≈50% mixes) and part2 (≈50% mixes)
3. **Run 2 Kaggle notebooks in parallel** → each processes its manifest part with resumability

## Phase 1: Create Training Manifest

### What It Does

Queries HuggingFace Hub (`Uday-4/djmix-v3`) for ALL mixes with residuals (even partial data) and creates a comprehensive manifest.

### Files

- **Input:** Reference manifest (e.g., `data/manifest_100mix.json`) for mix metadata
- **Output:** `data/manifest_training_all_available.json`
- **Script:** `scripts/create_training_manifest.py`

### Usage

```bash
# Run locally (with HF_TOKEN for private repo access)
export HF_TOKEN="hf_..."
python3 scripts/create_training_manifest.py \
  --output data/manifest_training_all_available.json \
  --reference data/manifest_100mix.json
```

### Output Example

```json
{
  "metadata": {
    "total_mixes": 65,
    "total_transitions": 1234,
    "total_residuals": 980,
    "mixes_with_partial_data": 12,
    "creation_date": "2026-03-13T19:05:12",
    "includes_partial_data": true,
    "source": "HuggingFace Hub (Uday-4/djmix-v3)"
  },
  "mixes": [
    {
      "id": "mix0021",
      "usable_transitions": 15,
      "num_residuals": 14,
      "has_complete_stems": true,
      "title": "2013-07-17 - Ulterior Motive - Metalheadz, Rinse FM",
      "notes": "partial residuals"
    },
    ...
  ]
}
```

**Summary printed:**
```
======================================================================
Training Manifest Created
======================================================================
Total mixes:              65
Total transitions:        1234
Total residuals:          980
Mixes with partial data:  12
Creation date:            2026-03-13 19:05:12
Output:                   data/manifest_training_all_available.json
======================================================================
```

## Phase 2: Split Manifest Into 2 Parts

### What It Does

Splits the training manifest 50/50 by mix count, using round-robin assignment to balance transitions across both parts.

### Files

- **Input:** `data/manifest_training_all_available.json`
- **Output:**
  - `data/manifest_training_part1.json` (≈50% mixes, N transitions)
  - `data/manifest_training_part2.json` (≈50% mixes, M transitions)
- **Script:** `scripts/split_training_manifest.py`

### Usage

```bash
python3 scripts/split_training_manifest.py \
  data/manifest_training_all_available.json
```

### Output

```
Splitting manifest: data/manifest_training_all_available.json
================================================================================
Part 1:  32 mixes |  617 transitions |  490 residuals | data/manifest_training_part1.json
Part 2:  33 mixes |  617 transitions |  490 residuals | data/manifest_training_part2.json
================================================================================
```

**Result:** Two balanced manifests with ~617 transitions each, ready for parallel processing.

## Phase 3: Run Step 7 on Kaggle (2 Parallel Notebooks)

### Files

- **Notebook:** `kaggle_phaseB_step7_resumable.ipynb`
- **Upload to:** Kaggle Datasets or Notebooks

### Key Features

1. **Resumable:** Saves progress after each mix to both:
   - Local file (survives cell restart)
   - HuggingFace Hub (survives session timeout)

2. **Parallel-safe:** Two notebooks can run simultaneously:
   - Notebook 1: processes `manifest_training_part1`
   - Notebook 2: processes `manifest_training_part2`
   - No race conditions (each notebook writes its own progress file)

3. **Comprehensive logging:**
   ```
   [13:45:22] INFO: Starting Step 7 for manifest_training_part1
   [13:45:22] INFO: Found 32 mixes, 617 transitions pending
   [13:45:30] INFO: [1/32] mix0021: 15 transitions pending
   [13:45:45] INFO: [1/32] mix0021: ✓ 12 ok, 0 failed | ETA: 127 min
   [13:45:55] INFO: Progress checkpoint saved
   ...
   [14:52:00] INFO: ======== COMPLETE ========
   [14:52:00] INFO: Curves extracted: 512 transitions
   ```

4. **Dry-run support:** Set `DRY_RUN=True` in Cell 2 to test on 1 mix

### Setup Instructions

**On Kaggle:**

1. Upload v4 repo and ai-dj source to a Kaggle Dataset (or clone from GitHub)
2. Create 2 new Kaggle Notebooks, set Accelerator = None (CPU)
3. Copy `kaggle_phaseB_step7_resumable.ipynb` to each notebook

**Notebook 1:**
```python
# Cell 2 Configuration
MANIFEST = "manifest_training_part1"
DRY_RUN = False  # Set to True for quick test
```

**Notebook 2:**
```python
# Cell 2 Configuration
MANIFEST = "manifest_training_part2"
DRY_RUN = False
```

### Running the Notebooks

1. **Test mode (quick 1-mix test):**
   ```python
   DRY_RUN = True
   CHECKPOINT_FREQ = 1
   ```
   Run cells 1-6. Should complete in ~5 min.

2. **Full run:**
   ```python
   DRY_RUN = False
   CHECKPOINT_FREQ = 5
   ```
   Run cells 1-6. Expected duration: 3-5 hours per notebook.

3. **Resume after timeout:**
   - If Kaggle session dies after 12h, restart the notebook
   - Run cells 1-6 again
   - It will automatically resume from the last saved progress (no data loss)

### Progress Tracking

**During execution:**
```python
# Cell 6 shows current status
verify = verify_curves_extracted(DATA_ROOT, mixes)
print(f"Curves extracted: {verify['total']}")
print(f"  (from {len(verify['by_mix'])} mixes)")
```

**Between sessions:**
```python
# Load progress from previous session
progress = json.load(open("manifest_training_part1_progress.json"))
print(f"Completed: {len(progress['completed_mixes'])} mixes")
print(f"Total ok: {progress['stats']['total_ok']}")
print(f"Total failed: {progress['stats']['total_failed']}")
```

## Notebook Cell Reference

### Cell 1: Setup & Imports
- Installs AIDJ dependencies
- Imports core modules (StemCurveExtractor, StemCache, etc.)

### Cell 2: Configuration
- `MANIFEST`: which part to process (`"manifest_training_part1"` or `"manifest_training_part2"`)
- `CURVE_WORKERS`: parallel workers (default: 4)
- `CHECKPOINT_FREQ`: save progress every N mixes (default: 5)
- `DRY_RUN`: test mode (default: False)

### Cell 3: Comprehensive Logger
- Custom logger with time-stamped output
- Methods: `info()`, `warning()`, `error()`, `mix_start()`, `mix_complete()`, `section()`, `summary()`

### Cell 4: Progress Tracking & Resumability
- Loads manifest
- Loads local progress (from last session)
- Identifies pending mixes
- If DRY_RUN: limit to 1 mix

### Cell 5: Curve Extraction Loop
- Main processing loop
- For each mix: extract curves for all transitions
- Save progress after each mix
- Calculate ETA based on elapsed time
- Checkpoint every N mixes

### Cell 6: Verification & Status Check
- Count total curves extracted
- Show sample mix statistics
- Confirm readiness for Phase C training

## Resumability Details

### How It Works

After processing each mix, the notebook saves progress to:

1. **Local file** (for session restart): `manifest_training_part1_progress.json`
   ```json
   {
     "completed_mixes": {
       "mix0021": {"num_ok": 12, "num_failed": 0, "completed_at": "2026-03-13T13:45:55"},
       "mix0022": {"num_ok": 14, "num_failed": 1, "completed_at": "2026-03-13T13:46:30"},
       ...
     },
     "stats": {"total_ok": 512, "total_failed": 3}
   }
   ```

2. **HuggingFace Hub** (for session timeout): uploads to `Uday-4/djmix-v3`
   - Same file, stored remotely for safety

### If Session Dies

1. **Restart the notebook** (after 12h timeout or manual stop)
2. **Run cells 1-6 again**
3. Cell 4 will:
   - Load local progress from last session
   - Load HF progress (if available)
   - Merge both: `completed = local ∪ HF`
   - Only process `pending_mixes = all_mixes - completed`
4. **No data loss** — already extracted curves are skipped, new ones continue

## Common Patterns

### Check Progress During Execution

```python
# In Cell 5, after a few mixes have completed
progress = json.load(open("manifest_training_part1_progress.json"))
completed = len(progress["completed_mixes"])
total = len(mixes)
pct = 100 * completed / total
print(f"Progress: {completed}/{total} ({pct:.1f}%)")
print(f"Curves so far: {progress['stats']['total_ok']}")
```

### View All Progress Files on HF

```python
# After both notebooks complete
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_tree("Uday-4/djmix-v3", token=HF_TOKEN, recursive=True)
progress_files = [f for f in files if "phase_b_progress_step7" in f.path]
print(f"Progress files: {progress_files}")
```

### Combine Results from Both Notebooks

```python
# After both notebooks complete, combine results
import json

prog1 = json.load(open("manifest_training_part1_progress.json"))
prog2 = json.load(open("manifest_training_part2_progress.json"))

total_ok = prog1["stats"]["total_ok"] + prog2["stats"]["total_ok"]
total_failed = prog1["stats"]["total_failed"] + prog2["stats"]["total_failed"]
total_mixes = len(prog1["completed_mixes"]) + len(prog2["completed_mixes"])

print(f"Combined Results:")
print(f"  Mixes processed: {total_mixes}")
print(f"  Curves extracted: {total_ok}")
print(f"  Failed: {total_failed}")
print(f"  Ready for Phase C training!")
```

## Troubleshooting

### "HF_TOKEN not set" warning

This is OK if the repository is public or you're only reading local data. For uploading progress to HF, set the token:

```python
import os
os.environ["HF_TOKEN"] = "hf_..."
```

### "missing stems" error

Some transitions don't have cached stems. This is expected and counted in `total_failed`. The extraction continues with other transitions.

### Timeout after 12h

This is expected on Kaggle free tier. The notebook is designed to handle this:
1. Progress is saved every 5 mixes (by default)
2. On restart, it automatically resumes from the last saved checkpoint
3. No need to manually do anything — just restart the notebook

### Memory issues

If the notebook runs out of memory:
1. Reduce `CURVE_WORKERS` (default: 4 → try 2)
2. Reduce `CHECKPOINT_FREQ` (default: 5 → try 3)
3. Clear cell outputs: Kernel → Restart & Clear Output

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/create_training_manifest.py` | Query HF for all available residuals, create training manifest |
| `scripts/split_training_manifest.py` | Split manifest 50/50 for parallel processing |
| `kaggle_phaseB_step7_resumable.ipynb` | Main Step 7 notebook (run 2x, one per manifest part) |
| `data/manifest_training_all_available.json` | Output from Phase 1 (all available mixes) |
| `data/manifest_training_part1.json` | Output from Phase 2 (first 50% of mixes) |
| `data/manifest_training_part2.json` | Output from Phase 2 (second 50% of mixes) |
| `manifest_training_part1_progress.json` | Notebook 1 progress (local + HF) |
| `manifest_training_part2_progress.json` | Notebook 2 progress (local + HF) |

## Next Steps

After Step 7 completes (both notebooks):

1. **Verify results:**
   ```bash
   # Check total curves extracted
   find ~/djmix/results/stem_curves -name "*.npz" | wc -l
   ```

2. **Proceed to Phase C (Training):**
   - Run `kaggle_phase_c.ipynb`
   - Build dataset from curves
   - Train StemTransitionNet

## Questions?

Refer to the main `CLAUDE.md` for overall pipeline architecture and Phase A/B/C overview.
