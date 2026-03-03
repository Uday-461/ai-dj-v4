#!/bin/bash
# One-time setup for hp-mint (Ryzen 5 5600U, 14GB RAM)
set -e

echo "=== Installing system dependencies ==="
sudo apt-get update && sudo apt-get install -y ffmpeg git

echo "=== Creating Python virtual environment ==="
python3 -m venv ~/aidj-venv
source ~/aidj-venv/bin/activate

echo "=== Installing PyTorch (CPU) ==="
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing AI DJ dependencies ==="
pip install demucs librosa soundfile cvxpy scipy numpy scikit-image
pip install huggingface_hub yt-dlp apify-client
pip install pytsmod pyloudnorm

echo "=== Cloning and installing project ==="
if [ ! -d ~/ai-dj ]; then
    echo "Please clone the repo to ~/ai-dj manually"
else
    cd ~/ai-dj/v3
    pip install -e .
fi

echo "=== Setup complete ==="
echo "Activate with: source ~/aidj-venv/bin/activate"
echo "Run pipeline:  cd ~/ai-dj/v3 && python hp_pipeline.py --manifest data/manifest_2mix.json"
