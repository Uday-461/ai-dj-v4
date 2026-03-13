# Kaggle Setup: How Notebooks Access Training Manifests

## Overview

The Step 7 notebooks need access to the training manifests and the v4 codebase. There are **3 options** for getting manifests to Kaggle, with different trade-offs:

| Option | Method | Setup | Update Manifest | Pros | Cons |
|--------|--------|-------|-----------------|------|------|
| **A** | GitHub (recommended) | Push v4 to GitHub, clone in notebook | Git pull | One source of truth, easy to update | Requires GitHub push |
| **B** | Kaggle Dataset | Upload v4 as Dataset, version it | Re-upload dataset | Isolated, no external dependency | Manual re-upload on changes |
| **C** | HF Hub | Upload manifests to `Uday-4/djmix-v3` | Automatic (script does it) | Decoupled from code, fast push | Separate upload pipeline |

**Recommended: Option A (GitHub) + Option C (HF as backup)**
- Push v4 to GitHub (contains manifests)
- In notebook: `git clone` the repo
- Manifests are in the code, always in sync
- Optional: also push progress to HF (already in notebook)

---

## Option A: GitHub (Recommended)

### Setup (One-Time)

1. **Create GitHub repo for v4**
   ```bash
   cd /Users/uday/code/ai-dj/v4
   git init
   git add .
   git commit -m "Initial v4 commit"
   git remote add origin https://github.com/Uday-461/ai-dj-v4.git
   git push -u origin main
   ```

2. **Make repo public** (or ensure Kaggle can access if private)
   - On GitHub: Settings → Visibility → Make public

### In Kaggle Notebook

```python
# Cell 1 (added before other imports)
import subprocess
import os

# Clone v4 repo
if not os.path.exists('/kaggle/working/v4'):
    subprocess.run(
        ['git', 'clone', 'https://github.com/Uday-461/ai-dj-v4.git', '/kaggle/working/v4'],
        check=True
    )

sys.path.insert(0, '/kaggle/working/v4')
```

Then in Cell 2:
```python
MANIFEST_PATH = Path('/kaggle/working/v4/data/manifest_training_part1.json')
```

### Update Manifests

After running Phase 2 locally:
```bash
cd /Users/uday/code/ai-dj/v4
git add data/manifest_training*.json
git commit -m "Update training manifests"
git push
```

Then in Kaggle, just restart notebook (git clone gets latest).

---

## Option B: Kaggle Dataset (Isolated)

### Setup (One-Time)

1. **Create Kaggle Dataset from v4**
   ```bash
   # Install kaggle CLI
   pip install kaggle

   # Create dataset
   cd /Users/uday/code/ai-dj
   kaggle datasets create \
     --dir-mode zip \
     --path v4 \
     --dataset-slug ai-dj-v4 \
     --title "AI DJ v4"
   ```

2. **Note the dataset version number** (e.g., `v1`)

### In Kaggle Notebook

The manifests are already available as an input dataset:

```python
# Cell 1
MANIFEST_PATH = Path('/kaggle/input/ai-dj-v4/data/manifest_training_part1.json')
```

No cloning needed!

### Update Manifests

After running Phase 2 locally:
```bash
# Update the dataset (creates new version)
kaggle datasets version \
  --dir-mode zip \
  --path /Users/uday/code/ai-dj/v4 \
  --dataset-slug ai-dj-v4 \
  --version-notes "Updated training manifests"
```

Then in Kaggle:
- When creating notebook, attach `ai-dj-v4` as input
- Kaggle will use the latest version
- No changes needed in notebook code

---

## Option C: HuggingFace Hub (Backup)

### Concept

Upload manifests to the same HF repo where we store residuals/stems:

```
Uday-4/djmix-v3/
  ├── tracks/
  ├── mix_segments/
  ├── stems/
  ├── results/
  └── manifests/                    ← NEW
      ├── manifest_training_all_available.json
      ├── manifest_training_part1.json
      └── manifest_training_part2.json
```

### Setup (One-Time)

```bash
# Local: upload manifests to HF
python3 << 'EOF'
from huggingface_hub import HfApi
import os

api = HfApi()
hf_token = os.environ.get("HF_TOKEN")

for file in [
    "data/manifest_training_all_available.json",
    "data/manifest_training_part1.json",
    "data/manifest_training_part2.json"
]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=f"manifests/{Path(file).name}",
        repo_id="Uday-4/djmix-v3",
        token=hf_token
    )
    print(f"Uploaded {file} to HF")
EOF
```

### In Kaggle Notebook

```python
# Cell 2 (if using HF as backup)
from huggingface_hub import hf_hub_download

# Download from HF if not available locally
HF_MANIFEST = "manifest_training_part1.json"
MANIFEST_PATH = Path(f"/kaggle/working/{HF_MANIFEST}")

if not MANIFEST_PATH.exists():
    hf_hub_download(
        repo_id="Uday-4/djmix-v3",
        filename=f"manifests/{HF_MANIFEST}",
        cache_dir="/kaggle/working",
        token=HF_TOKEN
    )
```

### Update Manifests

Automatic! After Phase 2 (split_training_manifest.py), the script can upload:

```bash
# Modify split_training_manifest.py to add:
api = HfApi()
api.upload_file(
    path_or_fileobj="data/manifest_training_part1.json",
    path_in_repo="manifests/manifest_training_part1.json",
    repo_id="Uday-4/djmix-v3",
    token=hf_token
)
```

---

## Recommended Setup: A + C (GitHub + HF Backup)

### Why?

1. **GitHub** (Primary)
   - v4 code + manifests always in sync
   - Easy to version and review
   - Trivial to update (just `git push`)
   - Free hosting

2. **HF Hub** (Backup)
   - Manifests also available on HF
   - Useful if GitHub is ever down
   - Already in use for data/progress
   - Minimal extra work

### Implementation

**Local (hp-mint or Mac):**
```bash
# After Phase 2 (split manifests)

# 1. Push to GitHub
cd /Users/uday/code/ai-dj/v4
git add data/manifest_training*.json
git commit -m "Update training manifests (Phase 2)"
git push

# 2. Also upload to HF (optional, for backup)
export HF_TOKEN="hf_..."
python3 scripts/upload_manifests_to_hf.py  # (new helper script)
```

**Kaggle Notebook (Cell 1):**
```python
import subprocess
import sys
from pathlib import Path

# Clone v4 repo (includes manifests)
if not Path('/kaggle/working/v4').exists():
    subprocess.run([
        'git', 'clone',
        'https://github.com/Uday-461/ai-dj-v4.git',
        '/kaggle/working/v4'
    ], check=True)

sys.path.insert(0, '/kaggle/working/v4')
```

**Kaggle Notebook (Cell 2):**
```python
MANIFEST_PATH = Path('/kaggle/working/v4/data/manifest_training_part1.json')
```

---

## Detailed Comparison

### Option A: GitHub

**Pros:**
- ✓ Version control (git history)
- ✓ Easy to review/audit manifests
- ✓ One-line update (git push)
- ✓ Free, reliable
- ✓ v4 code always in sync with manifests

**Cons:**
- ✗ Requires GitHub account & push access
- ✗ Network dependency (clone on each Kaggle run)
- ✗ Slightly slower (network clone)

**Best for:** Primary source, easy to track changes

---

### Option B: Kaggle Dataset

**Pros:**
- ✓ Built into Kaggle (no external dependency)
- ✓ Fast access (local dataset)
- ✓ Versioning built-in
- ✓ No cloning needed (already on disk)

**Cons:**
- ✗ Manual re-upload on manifest changes
- ✗ Slower to update (re-upload dataset)
- ✗ Duplicates code (also have v4 repo elsewhere)

**Best for:** Static data, infrequent updates

---

### Option C: HuggingFace Hub

**Pros:**
- ✓ Already using HF for data
- ✓ Single upload point
- ✓ Decoupled from code repo
- ✓ Works with existing progress tracking

**Cons:**
- ✗ Requires HF_TOKEN
- ✗ Network dependency (download on Kaggle)
- ✗ Not version controlled like GitHub

**Best for:** Backup/fallback, decoupled data

---

## Current Status (Phase 2 Complete)

✅ Manifests created:
- `data/manifest_training_all_available.json` (100 mixes, 2931 transitions)
- `data/manifest_training_all_available_part1.json` (50 mixes, 1473 transitions)
- `data/manifest_training_all_available_part2.json` (50 mixes, 1458 transitions)

### Next Steps

1. **Create GitHub repo** (Option A):
   ```bash
   cd /Users/uday/code/ai-dj/v4
   git init && git add . && git commit -m "Initial commit"
   git remote add origin https://github.com/Uday-461/ai-dj-v4.git
   git push -u origin main
   ```

2. **Or create Kaggle Dataset** (Option B):
   ```bash
   kaggle datasets create --path v4 --dataset-slug ai-dj-v4
   ```

3. **Or upload to HF** (Option C):
   ```bash
   export HF_TOKEN="hf_..."
   python3 scripts/upload_manifests_to_hf.py
   ```

4. **Update notebook** with correct path (see examples above)

---

## Manifest Loading in Notebook

Regardless of which option, the notebook code is the same:

```python
# Cell 2
from pathlib import Path
import json

# Option A (GitHub):
MANIFEST_PATH = Path('/kaggle/working/v4/data/manifest_training_part1.json')

# Option B (Kaggle Dataset):
# MANIFEST_PATH = Path('/kaggle/input/ai-dj-v4/data/manifest_training_part1.json')

# Option C (HF Hub):
# MANIFEST_PATH = Path('/kaggle/working/manifest_training_part1.json')  (after download)

# Load manifest
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

print(f"Loaded {manifest['metadata']['total_mixes']} mixes")
```

---

## Recommendation Summary

**Use GitHub (Option A):**
1. Push v4 to GitHub
2. In Kaggle notebook: `git clone` the repo
3. Manifests are always in sync with code
4. Update process: just `git push` locally

**Backup plan: Also upload to HF (Option C)**
1. After Phase 2, upload manifests to HF Hub
2. Notebook can fall back to HF if GitHub is unavailable
3. Already integrated with progress tracking

**Why not Kaggle Dataset (Option B)?**
- Manual re-upload required on changes
- Slower update cycle
- Good for truly static data, but manifests may change

---

## Files to Keep in Sync

If using GitHub (recommended):
```
v4/
├── scripts/
│   ├── create_training_manifest.py
│   └── split_training_manifest.py
├── data/
│   ├── manifest_100mix.json              (reference)
│   ├── manifest_training_all_available.json
│   ├── manifest_training_all_available_part1.json  ← push these
│   └── manifest_training_all_available_part2.json
├── kaggle_phaseB_step7_resumable.ipynb
└── STEP7_GUIDE.md
```

After each Phase 2 run:
```bash
git add data/manifest_training*.json
git commit -m "Update training manifests"
git push
```

Kaggle notebook will automatically get the latest on next run (or git pull in notebook).
