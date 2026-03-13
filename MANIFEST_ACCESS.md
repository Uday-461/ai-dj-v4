# How Kaggle Notebooks Access Training Manifests

## Current Status

Phase 1 & 2 are complete:
- ✅ `data/manifest_training_all_available.json` (100 mixes, 2931 transitions)
- ✅ `data/manifest_training_all_available_part1.json` (50 mixes, 1473 transitions)
- ✅ `data/manifest_training_all_available_part2.json` (50 mixes, 1458 transitions)

Now: **How do we get these manifests to Kaggle for Step 7?**

---

## 3 Options Explained

### **Option A: GitHub (Recommended) ✨ Best for Active Development**

The notebook clones the v4 repo from GitHub, so manifests are always in sync with code.

**Setup:**
```bash
# One-time: push v4 to GitHub
cd /Users/uday/code/ai-dj/v4
git init
git add .
git commit -m "Initial v4 commit with training manifests"
git remote add origin https://github.com/Uday-461/ai-dj-v4.git
git push -u origin main
```

**In Kaggle Notebook (Cell 1):**
```python
# Automatically done by the notebook
V4_PATH = Path('/kaggle/working/v4')
if not V4_PATH.exists():
    subprocess.run([
        'git', 'clone',
        'https://github.com/Uday-461/ai-dj-v4.git',
        str(V4_PATH)
    ], check=True)
```

**To update manifests:**
```bash
# After running Phase 2 locally:
git add data/manifest_training*.json
git commit -m "Update training manifests"
git push
```

Then restart the Kaggle notebook (git clone gets latest).

**Pros:**
- ✅ Version control (git history of manifest changes)
- ✅ One source of truth (code + manifests together)
- ✅ Easy to update (just git push)
- ✅ Free and reliable
- ✅ Notebook automatically fetches latest on restart

**Cons:**
- ❌ Network dependency (clone each time)
- ❌ Slightly slower (few seconds for git clone)
- ❌ Requires GitHub account

---

### **Option B: Kaggle Dataset (Best for Static Data)**

You create a Kaggle Dataset from v4, then attach it to the notebook.

**Setup:**
```bash
# One-time: create dataset
kaggle datasets create \
  --dir-mode zip \
  --path /Users/uday/code/ai-dj/v4 \
  --dataset-slug ai-dj-v4 \
  --title "AI DJ v4"
```

**In Kaggle Notebook (Cell 1):**
```python
# Option B: Use pre-attached dataset (instead of git clone)
V4_PATH = Path('/kaggle/input/ai-dj-v4')
```

**Attach dataset in Kaggle:**
- When creating notebook, add "ai-dj-v4" as input dataset
- It's immediately available as `/kaggle/input/ai-dj-v4`

**To update manifests:**
```bash
# After running Phase 2 locally:
kaggle datasets version \
  --dir-mode zip \
  --path /Users/uday/code/ai-dj/v4 \
  --dataset-slug ai-dj-v4 \
  --version-notes "Updated training manifests"
```

Then in Kaggle notebook, the dataset auto-updates to latest version.

**Pros:**
- ✅ Fast (data already on disk, no clone/download)
- ✅ Isolated from external sources
- ✅ Versioning built-in
- ✅ No need to clone each run

**Cons:**
- ❌ Manual re-upload on manifest changes
- ❌ Slower update cycle (need to version dataset)
- ❌ Duplicates data (code in multiple places)

---

### **Option C: HuggingFace Hub (Best for Decoupled Data)**

Upload manifests to the same HF repo where residuals live.

**Setup:**
```bash
# One-time: upload to HF
python3 << 'EOF'
from huggingface_hub import HfApi
import os

api = HfApi()
hf_token = os.environ.get("HF_TOKEN")

for file in [
    "data/manifest_training_all_available.json",
    "data/manifest_training_all_available_part1.json",
    "data/manifest_training_all_available_part2.json"
]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=f"manifests/{Path(file).name}",
        repo_id="Uday-4/djmix-v3",
        token=hf_token
    )
EOF
```

**In Kaggle Notebook (Cell 2):**
```python
# Download from HF if not available locally
from huggingface_hub import hf_hub_download

HF_MANIFEST = "manifest_training_all_available_part1.json"
MANIFEST_PATH = Path("/kaggle/working/") / HF_MANIFEST

if not MANIFEST_PATH.exists():
    hf_hub_download(
        repo_id="Uday-4/djmix-v3",
        filename=f"manifests/{HF_MANIFEST}",
        cache_dir="/kaggle/working",
        token=HF_TOKEN
    )
```

**To update manifests:**
```bash
# Just re-run the upload script after Phase 2
export HF_TOKEN="hf_..."
python3 scripts/upload_manifests_to_hf.py
```

**Pros:**
- ✅ Decoupled from code (data lives with data)
- ✅ Already using HF for residuals
- ✅ Automatic sync (just re-run script)
- ✅ Good backup if GitHub unavailable

**Cons:**
- ❌ Requires HF_TOKEN
- ❌ Network dependency (download each run)
- ❌ Extra upload step (not automatic)

---

## Recommendation: **Option A + Option C**

### Why This Combination?

1. **Option A (GitHub)** as primary:
   - v4 code + manifests always in sync
   - Easy to update (just git push)
   - One source of truth
   - No extra tokens needed

2. **Option C (HF)** as backup:
   - Manifests also on HF
   - Useful if GitHub ever unavailable
   - Already integrated with progress tracking
   - Minimal extra work

### Implementation (Recommended)

**Local (hp-mint or Mac):**

```bash
# Step 1: Push to GitHub (one-time setup)
cd /Users/uday/code/ai-dj/v4
git init && git add . && git commit -m "Initial v4"
git remote add origin https://github.com/Uday-461/ai-dj-v4.git
git push -u origin main

# Step 2: After Phase 2, update both sources
python3 scripts/split_training_manifest.py data/manifest_training_all_available.json

git add data/manifest_training*.json
git commit -m "Update training manifests"
git push

export HF_TOKEN="hf_..."
# Also upload to HF as backup (script below)
python3 << 'SCRIPT'
from huggingface_hub import HfApi
from pathlib import Path
import os

hf_token = os.environ.get("HF_TOKEN")
api = HfApi()
for file in ["data/manifest_training_all_available.json",
             "data/manifest_training_all_available_part1.json",
             "data/manifest_training_all_available_part2.json"]:
    api.upload_file(file, f"manifests/{Path(file).name}",
                    "Uday-4/djmix-v3", hf_token)
SCRIPT
```

**Kaggle Notebook (no changes needed):**
The updated notebook already supports Option A:
- Cell 1 clones from GitHub
- Cell 2 loads manifests from cloned repo
- Works out of the box!

---

## Current Notebook Status

The notebook has been updated to support **Option A (GitHub)**:

**Cell 1:** Clones v4 from GitHub (or uses pre-attached dataset)
```python
V4_PATH = Path('/kaggle/working/v4')
if not V4_PATH.exists():
    subprocess.run(['git', 'clone', 'https://github.com/Uday-461/ai-dj-v4.git', str(V4_PATH)])
```

**Cell 2:** Loads manifest from V4_PATH
```python
MANIFEST = "manifest_training_all_available_part1"
MANIFEST_PATH = V4_PATH / "data" / f"{MANIFEST}.json"
```

---

## Quick Start

### For GitHub (Recommended)

1. **Push v4 to GitHub:**
   ```bash
   cd /Users/uday/code/ai-dj/v4
   git init
   git add .
   git commit -m "Initial v4"
   git remote add origin https://github.com/Uday-461/ai-dj-v4.git
   git push -u origin main
   ```

2. **In Kaggle:** Attach notebook with no input datasets (it clones repo)

3. **Run:** Cells 1-6 work automatically

4. **Update manifests:**
   ```bash
   # After Phase 2
   git add data/manifest_training*.json
   git commit -m "Update manifests"
   git push
   # Restart Kaggle notebook (gets latest)
   ```

### For Kaggle Dataset (Alternative)

1. **Create dataset:**
   ```bash
   kaggle datasets create --path v4 --dataset-slug ai-dj-v4
   ```

2. **In Kaggle:** Attach "ai-dj-v4" dataset as input

3. **In Cell 1:** Comment out GitHub clone, use:
   ```python
   V4_PATH = Path('/kaggle/input/ai-dj-v4')
   ```

4. **Run:** Cells 1-6 work automatically

---

## Summary Table

| Aspect | GitHub (A) | Kaggle Dataset (B) | HF Hub (C) |
|--------|-----------|-------------------|-----------|
| Setup | git init, git push | kaggle datasets create | hf_hub_download |
| Access time | 3-5 sec (clone) | Instant (pre-attached) | 2-3 sec (download) |
| Update | git push | kaggle datasets version | Re-run upload script |
| Source of truth | Code repo | Separate dataset | HF data repo |
| Versioning | Git history | Dataset versions | No versioning |
| Recommendation | ⭐⭐⭐ Primary | ⭐⭐ If isolated | ⭐⭐ Backup |

---

## Conclusion

**Use GitHub (Option A)** as primary:
1. Easy one-line update (git push)
2. Code + manifests always in sync
3. Version control history
4. One source of truth

**Optionally also use HF (Option C)** as backup:
1. Already uploading progress there
2. Adds ~2 min to update process
3. Insurance if GitHub unavailable

The updated notebook supports both options in Cell 1 - just pick one!
