# Step 7 Quick Reference

## One-Line Commands

### Phase 1: Create Training Manifest
```bash
export HF_TOKEN="hf_..." && python3 scripts/create_training_manifest.py
```

### Phase 2: Split Manifest
```bash
cd /Users/uday/code/ai-dj/v4 && python3 scripts/split_training_manifest.py data/manifest_training_all_available.json
```

## Kaggle Notebook Setup

**Notebook 1:** Copy this to Cell 2:
```python
MANIFEST = "manifest_training_part1"
CURVE_WORKERS = 4
CHECKPOINT_FREQ = 5
DRY_RUN = False
```

**Notebook 2:** Copy this to Cell 2:
```python
MANIFEST = "manifest_training_part2"
CURVE_WORKERS = 4
CHECKPOINT_FREQ = 5
DRY_RUN = False
```

## Quick Test (5 min dry run)
```python
# In Cell 2, set:
DRY_RUN = True
CHECKPOINT_FREQ = 1
# Then run cells 1-6
```

## Resume After Timeout
1. Session dies after 12h ✓ (expected)
2. Restart Kaggle notebook
3. Run cells 1-6 again
4. It auto-resumes from last saved mix (no data loss)

## Check Progress
```python
# In Cell 6 or new cell
import json
progress = json.load(open("manifest_training_part1_progress.json"))
print(f"Completed: {len(progress['completed_mixes'])} mixes")
print(f"Curves: {progress['stats']['total_ok']}")
```

## Combine Results After Both Notebooks
```python
import json

p1 = json.load(open("manifest_training_part1_progress.json"))
p2 = json.load(open("manifest_training_part2_progress.json"))

total = p1["stats"]["total_ok"] + p2["stats"]["total_ok"]
print(f"Total curves: {total} (ready for Phase C training!)")
```

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/create_training_manifest.py` | Create manifest from HF |
| `scripts/split_training_manifest.py` | Split 50/50 for parallel |
| `kaggle_phaseB_step7_resumable.ipynb` | Main notebook (run 2x) |
| `data/manifest_training_all_available.json` | Output from Phase 1 |
| `data/manifest_training_part1.json` | Half 1 (Notebook 1) |
| `data/manifest_training_part2.json` | Half 2 (Notebook 2) |

## Time Estimates

- Phase 1: 10 sec
- Phase 2: 1 sec
- Phase 3 per notebook: 3-5 hours
- Total: 3-5 hours (parallel)

## Expected Output

After both notebooks complete:
```
Part 1: 617 curves ✓
Part 2: 617 curves ✓
Total: 1234 curves
Ready for Phase C training!
```

## Troubleshooting

### Memory warning?
- Reduce CURVE_WORKERS: 4 → 2
- Reduce CHECKPOINT_FREQ: 5 → 3

### Session timeout?
- No problem! Progress is saved.
- Just restart the notebook.

### HF_TOKEN warning?
- OK if just reading local data.
- For HF upload: `export HF_TOKEN="hf_..."`

### Missing stems error?
- Normal (some transitions lack stems).
- Counted in `total_failed`.
- Extraction continues.

## More Info

- Detailed guide: `STEP7_GUIDE.md`
- Architecture: `IMPLEMENTATION_SUMMARY.md`
- Pipeline overview: `CLAUDE.md`
