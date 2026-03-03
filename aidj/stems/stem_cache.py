from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from aidj import config

log = logging.getLogger(__name__)


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


class StemCache:
    """Disk cache for separated stems.

    Cache structure:
        {cache_root}/stems/{audio_type}/{audio_id}/{stem_name}.{ext}

    Where audio_type is "mixes" or "tracks".
    Supports OGG (default), FLAC, and WAV formats with automatic
    fallback to .wav on read if the configured format isn't found.
    """

    def __init__(self, cache_root: str | Path = None, sr: int = config.SR):
        self.cache_root = Path(cache_root or config.DATA_ROOT)
        self.sr = sr
        self._ext = config.STEM_EXT[config.STEM_FORMAT]

    def _stem_dir(self, audio_type: str, audio_id: str) -> Path:
        return self.cache_root / "stems" / audio_type / audio_id

    def _stem_path(self, audio_type: str, audio_id: str, stem: str) -> Path:
        return self._stem_dir(audio_type, audio_id) / f"{stem}{self._ext}"

    def _stem_path_with_fallback(self, audio_type: str, audio_id: str, stem: str) -> Path | None:
        """Return the path to an existing stem file, trying configured format first, then .wav fallback."""
        primary = self._stem_path(audio_type, audio_id, stem)
        if primary.exists():
            return primary
        if self._ext != ".wav":
            wav_path = self._stem_dir(audio_type, audio_id) / f"{stem}.wav"
            if wav_path.exists():
                return wav_path
        return None

    def has_stems(self, audio_type: str, audio_id: str) -> bool:
        """Check if all stems are cached for this audio."""
        stem_dir = self._stem_dir(audio_type, audio_id)
        if not stem_dir.exists():
            return False
        return all(
            self._stem_path_with_fallback(audio_type, audio_id, s) is not None
            for s in config.STEMS
        )

    def load_stems(self, audio_type: str, audio_id: str) -> dict[str, np.ndarray] | None:
        """Load cached stems.

        Returns dict mapping stem name -> numpy array, or None if not cached.
        Tries configured format first, falls back to .wav.
        """
        if not self.has_stems(audio_type, audio_id):
            return None

        result = {}
        for stem in config.STEMS:
            path = self._stem_path_with_fallback(audio_type, audio_id, stem)
            data, sr = sf.read(str(path))
            result[stem] = data

        return result

    def _write_stem(self, path: Path, audio: np.ndarray):
        """Write a single stem file in the configured format."""
        audio_f32 = audio.astype(np.float32)
        if self._ext == ".ogg" and _has_ffmpeg():
            # Use ffmpeg for precise OGG quality control
            path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "f32le", "-ar", str(self.sr), "-ac", "1",
                "-i", "pipe:0",
                "-c:a", "libvorbis", "-q:a", "6",
                str(path),
            ]
            subprocess.run(cmd, input=audio_f32.tobytes(), check=True)
        elif self._ext == ".ogg":
            sf.write(str(path), audio_f32, self.sr, format="OGG", subtype="VORBIS")
        else:
            sf.write(str(path), audio_f32, self.sr)

    def save_stems(self, audio_type: str, audio_id: str,
                   stems: dict[str, np.ndarray]):
        """Save stems to cache."""
        stem_dir = self._stem_dir(audio_type, audio_id)
        stem_dir.mkdir(parents=True, exist_ok=True)

        for stem_name, audio in stems.items():
            path = self._stem_path(audio_type, audio_id, stem_name)
            self._write_stem(path, audio)

        log.debug(f"Cached stems for {audio_type}/{audio_id}")

    def separate_and_cache(self, separator, audio_path: str,
                           audio_type: str, audio_id: str) -> dict[str, np.ndarray]:
        """Separate audio and cache results. Uses cache if available.

        Args:
            separator: StemSeparator instance
            audio_path: path to audio file
            audio_type: "mixes" or "tracks"
            audio_id: unique identifier for the audio

        Returns:
            dict mapping stem name -> numpy array
        """
        cached = self.load_stems(audio_type, audio_id)
        if cached is not None:
            return cached

        stems = separator.separate_mono(audio_path)
        self.save_stems(audio_type, audio_id, stems)
        return stems

    def separate_and_cache_segment(
        self, separator, audio: np.ndarray,
        audio_type: str, audio_id: str,
        sr: int = None,
    ) -> dict[str, np.ndarray]:
        """Separate an audio segment (numpy array) and cache results.

        Unlike separate_and_cache which takes a file path, this takes
        raw audio data — useful for mix transition segments that are
        extracted from the full mix audio.

        Args:
            separator: StemSeparator instance
            audio: 1D numpy array of audio samples
            audio_type: cache category (e.g., "mix_segments")
            audio_id: unique identifier for this segment
            sr: sample rate (defaults to self.sr)

        Returns:
            dict mapping stem name -> 1D numpy array
        """
        cached = self.load_stems(audio_type, audio_id)
        if cached is not None:
            return cached

        import tempfile

        sr = sr or self.sr
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            sf.write(tmp.name, audio.astype(np.float32), sr)
            stems = separator.separate_mono(tmp.name)

        self.save_stems(audio_type, audio_id, stems)
        return stems
