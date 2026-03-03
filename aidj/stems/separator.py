from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from aidj import config

log = logging.getLogger(__name__)


class StemSeparator:
    """Separate audio into stems using Demucs htdemucs model."""

    STEMS = config.STEMS  # ["drums", "bass", "vocals", "other"]

    def __init__(self, model_name: str = None, device: str | None = None):
        self.model_name = model_name or config.DEMUCS_MODEL
        self.device = self._get_device(device)
        self._model = None

    def _get_device(self, device=None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def model(self):
        if self._model is None:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels

            log.info(f"Loading Demucs model: {self.model_name}")
            self._model = get_model(self.model_name)
            if isinstance(self._model, BagOfModels):
                # For bag of models, each sub-model needs to be moved
                for m in self._model.models:
                    m.to(self.device)
            else:
                self._model.to(self.device)
            self._model.eval()
            log.info(f"Demucs model loaded on {self.device}")
        return self._model

    def separate(self, audio_path: str, sr: int = config.SR) -> dict[str, np.ndarray]:
        """Separate audio file into stems.

        Args:
            audio_path: path to audio file
            sr: target sample rate

        Returns:
            dict mapping stem name -> numpy array of shape (channels, samples)
        """
        import librosa
        from demucs.apply import apply_model

        # Load audio with librosa (avoids torchcodec dependency)
        raw, file_sr = librosa.load(audio_path, sr=None, mono=False)
        if raw.ndim == 1:
            raw = np.stack([raw, raw])  # mono -> stereo
        wav = torch.from_numpy(raw).float()

        # Resample if needed (Demucs expects 44100 Hz)
        if file_sr != self.model.samplerate:
            import torchaudio
            wav = torchaudio.transforms.Resample(file_sr, self.model.samplerate)(wav)

        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        # Add batch dimension: (batch, channels, samples)
        wav = wav.unsqueeze(0).to(self.device)

        # Apply model
        with torch.no_grad():
            sources = apply_model(self.model, wav, device=self.device)

        # sources shape: (batch, n_sources, channels, samples)
        sources = sources[0].cpu().detach().float().numpy()  # Remove batch dim

        # Map source indices to stem names
        # htdemucs order: drums, bass, other, vocals
        source_names = self.model.sources
        result = {}
        for i, name in enumerate(source_names):
            if name in self.STEMS:
                result[name] = sources[i]

        return result

    def separate_mono(self, audio_path: str, sr: int = config.SR) -> dict[str, np.ndarray]:
        """Separate audio and return mono stems.

        Returns:
            dict mapping stem name -> 1D numpy array
        """
        stereo_stems = self.separate(audio_path, sr)
        return {name: wav.mean(axis=0) for name, wav in stereo_stems.items()}
