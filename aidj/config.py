import os
from pathlib import Path

# --- Inherited from v1 ---
SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_TIME = 60  # seconds per transition window
CUE_BAR = 8

BAND_FREQS = [20, 300, 5000, 20000]

MAX_BPM_DIFF = 8
MAX_ENERGY_DIFF = 0.3

SCORE_WEIGHTS = {
    "bpm": 0.35,
    "key": 0.30,
    "energy": 0.20,
    "duration": 0.15,
}

CUE_OUT_PERCENT = 0.85
CUE_IN_PERCENT = 0.15
TARGET_LUFS = -14.0

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

# --- v2-specific ---
STEMS = ["drums", "bass", "vocals", "other"]
DEMUCS_MODEL = "htdemucs"

STEM_FORMAT = "ogg"  # "ogg", "flac", or "wav"
STEM_EXT = {"ogg": ".ogg", "flac": ".flac", "wav": ".wav"}

DATA_ROOT = Path(os.environ.get("AIDJ_DATA_ROOT", "~/djmix")).expanduser()

# Dataset JSON: check repo-local copy first, then fallback to reference_repos
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_PATHS = [
    _REPO_ROOT / "dataset" / "djmix-dataset.json",
    _REPO_ROOT.parent / "reference_repos" / "djmix-dataset" / "dataset" / "djmix-dataset.json",
]
DATASET_JSON = next((p for p in _DATASET_PATHS if p.exists()), _DATASET_PATHS[0])
SUBSET_SIZE = 500

# Sample rate for feature extraction (beat detection, chroma, MFCC).
# 22050Hz is sufficient for all perceptual features (max useful freq ~10kHz).
# Halves RAM and compute vs 44100Hz. config.SR stays 44100 for audio output.
FEATURE_SR = 22050

# Convex optimization params (from djmix-dataset)
OPT_SR = 44100
OPT_HOP = 16384  # 4x increase from 4096 to reduce CVXPY variable count
OPT_FFT = 16384  # must be >= OPT_HOP
OPT_MEL_BINS = 100
OPT_CQT_BINS = 100
OPT_FMIN = 20
OPT_FMAX = 20000

# EQ filter frequencies
EQ_CUTOFF_LOW = 180
EQ_CENTER_MID = 1000
EQ_CUTOFF_HIGH = 3000
EQ_MID_OPT_RANGE = (200, 5000)

# Transition length cap (seconds) — transitions longer than this are truncated
# before curve extraction to keep CVXPY solve times reasonable
MAX_TRANSITION_SECS = 60

# Model training
MODEL_N_MELS = 128
MODEL_N_STEMS = 4
MODEL_N_PARAMS = 8  # fader + eq_low/mid/high for prev and next
MODEL_IN_CHANNELS = 4  # prev, next, prev_residual, next_residual
MODEL_N_FFT = OPT_FFT
MODEL_HOP = OPT_HOP
