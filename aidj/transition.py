from __future__ import annotations

import logging
import os

import numpy as np
import torch
import librosa

from aidj import config
from aidj.stems.separator import StemSeparator
from aidj.mixer import mix_transition

log = logging.getLogger(__name__)


class TransitionGenerator:
    """v2 TransitionGenerator: stem-aware transitions using trained model.

    Same API as v1's TransitionGenerator for drop-in replacement.

    Pipeline:
    1. Extract windows around cue points
    2. Separate into stems via Demucs
    3. Compute per-stem mel spectrograms
    4. Predict per-stem curves via StemTransitionNet
    5. Apply curves to stems
    6. Sum stems -> transition audio
    7. Stitch: pre_a + transition + post_b
    """

    def __init__(self, model_path: str | None = None, device=None):
        self.device = self._get_device(device)
        self.separator = StemSeparator(device=str(self.device))
        self.model = None

        if model_path is not None and os.path.exists(model_path):
            from aidj.model.architecture import StemTransitionNet
            self.model = StemTransitionNet.load(model_path, device=self.device)
            log.info(f"Loaded StemTransitionNet from {model_path}")
        else:
            log.warning("No model path provided or model not found. "
                       "Will use simple crossfade fallback.")

    def _get_device(self, device=None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def generate(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        cue_a: float,
        cue_b: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a DJ transition between two tracks.

        Args:
            audio_a: mono audio of track A at 44100 Hz
            audio_b: mono audio of track B at 44100 Hz
            cue_a: cue-out point in track A (seconds)
            cue_b: cue-in point in track B (seconds)

        Returns:
            (transition_audio, full_mix_audio) as 1D float32 numpy arrays
        """
        if self.model is not None:
            try:
                return self._generate_with_model(audio_a, audio_b, cue_a, cue_b)
            except Exception as e:
                log.warning(f"Model-based generation failed ({e}), "
                           "falling back to crossfade")

        return self._generate_crossfade(audio_a, audio_b, cue_a, cue_b)

    def _generate_with_model(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        cue_a: float,
        cue_b: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate transition using the trained StemTransitionNet."""
        sr = config.SR
        window_sec = config.N_TIME
        window_samples = window_sec * sr
        half = window_samples // 2

        # Extract windows around cue points
        win_a = self._extract_window(audio_a, cue_a, sr, half, window_samples)
        win_b = self._extract_window(audio_b, cue_b, sr, half, window_samples)

        # Separate into stems
        import tempfile
        import soundfile as sf

        stems_a = self._separate_array(win_a, sr)
        stems_b = self._separate_array(win_b, sr)

        # Build model input: (1, 8, n_mels, n_frames)
        model_input = self._build_model_input(stems_a, stems_b)

        # Predict curves
        with torch.no_grad():
            pred = self.model(model_input)  # (1, 4, 8, n_frames)

        pred = pred[0].cpu().numpy()  # (4, 8, n_frames)

        # Convert predictions to curve dict
        curves = self._pred_to_curves(pred)

        # Apply curves to stems and mix
        transition = mix_transition(stems_a, stems_b, curves, sr)

        # Trim to window size
        transition = transition[:window_samples].astype(np.float32)

        # Stitch full mix
        cue_a_sample = int(cue_a * sr)
        cue_b_sample = int(cue_b * sr)
        pre_a = audio_a[:cue_a_sample]
        post_b = audio_b[cue_b_sample:]
        full_mix = np.concatenate([pre_a, transition, post_b]).astype(np.float32)

        return transition, full_mix

    def _generate_crossfade(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        cue_a: float,
        cue_b: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple stem-aware crossfade (no trained model needed)."""
        sr = config.SR
        window_sec = config.N_TIME
        window_samples = window_sec * sr
        half = window_samples // 2

        win_a = self._extract_window(audio_a, cue_a, sr, half, window_samples)
        win_b = self._extract_window(audio_b, cue_b, sr, half, window_samples)

        # Separate into stems
        stems_a = self._separate_array(win_a, sr)
        stems_b = self._separate_array(win_b, sr)

        # Simple per-stem crossfade
        n = window_samples
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)

        transition = np.zeros(n, dtype=np.float32)
        for stem in config.STEMS:
            a_stem = stems_a.get(stem, np.zeros(n))[:n]
            b_stem = stems_b.get(stem, np.zeros(n))[:n]

            # Pad if shorter than window
            if len(a_stem) < n:
                a_stem = np.pad(a_stem, (0, n - len(a_stem)))
            if len(b_stem) < n:
                b_stem = np.pad(b_stem, (0, n - len(b_stem)))

            transition += a_stem * fade_out + b_stem * fade_in

        # Stitch
        cue_a_sample = int(cue_a * sr)
        cue_b_sample = int(cue_b * sr)
        pre_a = audio_a[:cue_a_sample]
        post_b = audio_b[cue_b_sample:]
        full_mix = np.concatenate([pre_a, transition, post_b]).astype(np.float32)

        return transition, full_mix

    def _extract_window(self, audio, cue_sec, sr, half, window_samples):
        cue_sample = int(cue_sec * sr)
        start = max(0, cue_sample - half)
        end = start + window_samples
        if end > len(audio):
            end = len(audio)
            start = max(0, end - window_samples)
        window = audio[start:end]
        if len(window) < window_samples:
            window = np.pad(window, (0, window_samples - len(window)))
        return window.astype(np.float32)

    def _separate_array(self, audio, sr):
        """Separate a numpy array into stems using Demucs."""
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            sf.write(tmp.name, audio, sr)
            stems = self.separator.separate_mono(tmp.name)
        return stems

    def _build_model_input(self, stems_a, stems_b):
        """Build model input tensor from stem dicts.

        At inference we don't have the DJ's mix (we're generating it),
        so residual channels are filled with zeros.

        Returns (1, 16, n_mels, n_frames) tensor.
        """
        specs = []
        for stem in config.STEMS:
            a = stems_a.get(stem, np.zeros(1))
            b = stems_b.get(stem, np.zeros(1))

            spec_a = librosa.feature.melspectrogram(
                y=a, sr=config.SR, n_fft=config.OPT_FFT,
                hop_length=config.OPT_HOP, n_mels=config.MODEL_N_MELS,
                power=1,
            )
            spec_b = librosa.feature.melspectrogram(
                y=b, sr=config.SR, n_fft=config.OPT_FFT,
                hop_length=config.OPT_HOP, n_mels=config.MODEL_N_MELS,
                power=1,
            )

            # Normalize
            spec_a = librosa.amplitude_to_db(spec_a, ref=np.max)
            spec_b = librosa.amplitude_to_db(spec_b, ref=np.max)
            spec_a = (spec_a + 80) / 80  # normalize to ~[0, 1]
            spec_b = (spec_b + 80) / 80

            specs.append(spec_a)
            specs.append(spec_b)
            # Zero residual channels (no mix available at inference)
            specs.append(np.zeros_like(spec_a))
            specs.append(np.zeros_like(spec_b))

        # Ensure consistent frame count
        min_frames = min(s.shape[1] for s in specs)
        specs = [s[:, :min_frames] for s in specs]

        # Stack: (16, n_mels, n_frames)
        input_np = np.stack(specs)
        # Add batch: (1, 16, n_mels, n_frames)
        input_tensor = torch.from_numpy(input_np).float().unsqueeze(0)
        return input_tensor.to(self.device)

    def _pred_to_curves(self, pred):
        """Convert model prediction (4, 8, T) to curves dict."""
        curves = {}
        param_names = [
            'fader_prev', 'fader_next',
            'eq_prev_low', 'eq_prev_mid', 'eq_prev_high',
            'eq_next_low', 'eq_next_mid', 'eq_next_high',
        ]
        for s, stem in enumerate(config.STEMS):
            for p, param in enumerate(param_names):
                curves[f"{stem}_{param}"] = pred[s, p, :]
        return curves
