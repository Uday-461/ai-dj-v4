# AI DJ

A deep learning system that learns per-stem mixing curves from professional DJ mixes, then generates transitions between any two tracks.

## Motivation

My friends and I were tired of the music DJs played in clubs, but none of us were professional DJs. We wanted a "community DJ" — pick your songs, and the transitions happen automatically.

I've always loved music and had an interest in how it's made. I knew about spectrograms as data representations for audio and stem separation as a way to isolate instruments — I expanded on that knowledge and built this.

DJs create transitions by adjusting per-stem volume and EQ over time. This project learns those patterns from real DJ mixes using deep learning, so given any two tracks, it can predict how a professional DJ would transition between them.

## How It Works

The pipeline runs in four phases across different machines:

| Phase | What | Where |
|-------|------|-------|
| **A. Data Prep** | Download audio, detect beats, DTW alignment, extract transition regions | CPU machine |
| **B. Stems + Curves** | Demucs stem separation, CVXPY convex optimization for ground-truth curves | Kaggle GPU + CPU |
| **C. Training** | Build dataset, train StemTransitionNet, evaluate | Kaggle P100 |
| **D. Inference** | Separate new tracks, predict curves, apply per-stem mixing | Any machine |

### Ground Truth Extraction (Phases A-B)

To train the model, we need ground truth — what curves did the DJ actually use? The pipeline works backwards from finished DJ mixes:

1. **Beat detection** (BeatNet/librosa) finds beat positions in tracks and the full mix
2. **DTW alignment** matches each track's beats to the mix timeline, revealing where each track appears
3. **Transition extraction** identifies overlapping regions where two tracks play simultaneously
4. **Demucs separation** splits both the individual tracks and the mix segment into 4 stems (drums, bass, vocals, other)
5. **Residual computation** — `mix_stems - track_stems` captures what the DJ added/removed
6. **CVXPY optimization** solves for the per-stem fader and 3-band EQ curves that best reconstruct the mix from its component tracks

### Model Architecture

**StemTransitionNet** is a CNN + Transformer model with three components:

```
Input: mel spectrograms per stem (prev_track, next_track, prev_residual, next_residual)
  |
  v
[StemEncoder x4] -- one CNN per stem (drums, bass, vocals, other)
  |                  4-layer Conv2d, stride-2 on frequency axis
  |                  Input: (batch, 4_channels, 128_mels, T_frames)
  |                  Output: (batch, 256, T_frames)
  v
[CrossStemTransformer] -- 4 layers, 8 heads
  |                       lets stems coordinate timing
  |                       (e.g., drop bass while bringing in vocals)
  v
[CurveHead x4] -- one Conv1d decoder per stem
  |                outputs 8 params: fader_prev, fader_next,
  |                eq_low/mid/high for prev and next
  |                bounded [0, 2] via sigmoid
  v
Output: (batch, 4_stems, 8_params, T_frames)
```

The predicted curves are applied as time-varying fader gains and 3-band parametric EQ per stem. The two decks (outgoing + incoming track) are summed to produce the final transition audio.

**Loss function:** MAE reconstruction + monotonicity penalty (faders should fade in/out, not oscillate) + smoothness penalty (L1 on second derivative to prevent jitter).

### Inference

Given two tracks and their cue points:

1. Extract audio windows around the cue points
2. Separate each window into 4 stems with Demucs
3. Compute mel spectrograms (residual channels are zeroed — we don't have a DJ's mix to reference)
4. Feed through StemTransitionNet to predict 32 curves (4 stems x 8 params)
5. Apply fader + EQ curves to each stem
6. Sum stems, stitch with pre/post audio

## Dataset

Training data comes from the **DJ Mix Dataset** (Kim et al., DAFx 2022): 5,040 professional DJ mixes from mixesdb.com with metadata including tracklists, timestamps, and genre tags.

Production run: 100 mixes, ~1,400 tracks, ~1,100 transitions.

## Tech Stack

- **PyTorch** — model architecture and training
- **Demucs** (htdemucs) — stem separation into drums, bass, vocals, other
- **CVXPY** — convex optimization for ground-truth curve extraction
- **librosa** — mel spectrograms, audio features
- **BeatNet** — beat/downbeat detection
- **HuggingFace Hub** — dataset and model hosting
- **Kaggle** — free P100 GPU for training and Demucs

## Project Structure

```
v4/
  aidj/                        # core library
    model/
      architecture.py          # StemTransitionNet (CNN + Transformer)
      dataset.py               # PyTorch dataset for transition samples
      losses.py                # MAE + monotonicity + smoothness loss
      trainer.py               # training loop
    curves/
      optimizer.py             # CVXPY curve extraction
      eq_filters.py            # 3-band parametric EQ design
      stem_curve_extractor.py  # orchestrates per-transition extraction
    stems/
      separator.py             # Demucs wrapper
      stem_cache.py            # HF-backed stem cache
    data/
      downloader.py            # Apify + yt-dlp
      beat_detector.py         # BeatNet / librosa
      aligner.py               # DTW alignment
      transition_extractor.py  # transition region detection
      residual.py              # mix - track stem residuals
    mixer.py                   # per-stem curve application + summing
    transition.py              # end-to-end inference (TransitionGenerator)
    config.py                  # constants, paths, hyperparams
  scripts/01-11                # numbered pipeline scripts
  data/                        # manifests and dataset JSON
```

## Acknowledgments

This project builds on ideas and code from the following work:

**DJtransGAN** — Bo-Yu Chen et al., ICASSP 2022
- Paper: "Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks"
- Repo: [ChenPaulYu/DJtransGAN](https://github.com/ChenPaulYu/DJtransGAN) (MIT License)
- Inspiration for the differentiable fader/EQ approach. AI DJ extends this with per-stem curves and a CNN+Transformer architecture instead of a GAN.

**DJ Mix Dataset** — Taejun Kim et al., DAFx 2022
- Paper: "Joint Estimation of Fader and Equalizer Gains of DJ Mixers using Convex Optimization"
- Repo: [mir-aidj/djmix-dataset](https://github.com/mir-aidj/djmix-dataset)
- Provides the 5,040-mix dataset with metadata. The CVXPY curve extraction approach in this project references their convex optimization formulation.

**DJtransGAN-dg-pipeline** — Bo-Yu Chen
- Repo: [ChenPaulYu/DJtransGAN-dg-pipeline](https://github.com/ChenPaulYu/DJtransGAN-dg-pipeline) (MIT License)
- Data generation pipeline referenced for ground truth pair mining and alignment.

## License

MIT
