from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import librosa

from aidj import config
from aidj.curves.eq_filters import eq3_filters, bin_gains

log = logging.getLogger(__name__)


class OptConfig:
    """Configuration for the convex optimizer."""
    def __init__(
        self,
        sr=config.OPT_SR,
        hop=config.OPT_HOP,
        n_fft=config.OPT_FFT,
        num_mel_bins=config.OPT_MEL_BINS,
        fmin=config.OPT_FMIN,
        fmax=config.OPT_FMAX,
        cutoff_low=config.EQ_CUTOFF_LOW,
        center_mid=config.EQ_CENTER_MID,
        cutoff_high=config.EQ_CUTOFF_HIGH,
        mid_opt_range=config.EQ_MID_OPT_RANGE,
        spectrogram="mel",
    ):
        self.sr = sr
        self.hop = hop
        self.n_fft = n_fft
        self.num_mel_bins = num_mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.cutoff_low = cutoff_low
        self.center_mid = center_mid
        self.cutoff_high = cutoff_high
        self.mid_opt_range = mid_opt_range
        self.spectrogram = spectrogram


class EQ3FaderOptimizer:
    """Convex optimizer for fader + 3-band EQ curves.

    Given spectrograms of a DJ mix, previous track, and next track,
    solves for per-frame fader and EQ gains that best reconstruct
    the mix as a weighted sum of the two tracks.

    Port of djmix-dataset/djmix/cvxopt/optimizers/eq3_fader_optimizer.py.
    """

    def __init__(self, opt_config: OptConfig = None):
        c = opt_config or OptConfig()
        self.config = c

        # Build EQ filter bank
        self.filter_low, self.filter_mid, self.filter_high = eq3_filters(
            cutoff_low=c.cutoff_low,
            center_mid=c.center_mid,
            cutoff_high=c.cutoff_high,
            sr=c.sr,
        )

        # Frequency bins for the spectrogram type
        self.bin_freqs = librosa.mel_frequencies(c.num_mel_bins, fmin=c.fmin, fmax=c.fmax)

        # Per-band filter gains at each frequency bin
        self.bin_gains_dict = {
            'low': bin_gains(self.filter_low, self.bin_freqs, c.sr),
            'mid': bin_gains(self.filter_mid, self.bin_freqs, c.sr),
            'high': bin_gains(self.filter_high, self.bin_freqs, c.sr),
        }

        # Bin masks for each band
        self.bins = {
            'low': self.bin_freqs <= c.cutoff_low,
            'mid': (c.mid_opt_range[0] < self.bin_freqs) & (self.bin_freqs < c.mid_opt_range[1]),
            'high': c.cutoff_high <= self.bin_freqs,
        }

        # Band-specific gains (only bins within the band)
        self.band_gains = {
            band: self.bin_gains_dict[band][self.bins[band]]
            for band in ['low', 'mid', 'high']
        }

    def optimize(self, S_dj, S_prev, S_next, verbose=False):
        """Solve for fader + EQ curves.

        Args:
            S_dj: (n_bins, n_frames) spectrogram of DJ mix transition region
            S_prev: (n_bins, n_frames) spectrogram of previous track
            S_next: (n_bins, n_frames) spectrogram of next track
            verbose: whether to print solver output

        Returns:
            dict with keys like 'fader_prev', 'fader_next',
            'eq_prev_low', 'eq_prev_mid', 'eq_prev_high', etc.
        """
        num_frames = S_dj.shape[1]

        # Fader variables (monotonic: prev decreases, next increases)
        alpha_prev = cp.Variable(shape=num_frames, name='alpha_prev')
        alpha_next = cp.Variable(shape=num_frames, name='alpha_next')

        constraints = [
            alpha_prev >= 0,
            alpha_next >= 0,
            alpha_prev <= 2,
            alpha_next <= 2,
            cp.diff(alpha_prev) <= 0,
            cp.diff(alpha_next) >= 0,
        ]

        losses = {}

        for band in ['low', 'mid', 'high']:
            gamma_prev = cp.Variable(shape=num_frames, name=f'gamma_prev_{band}')
            gamma_next = cp.Variable(shape=num_frames, name=f'gamma_next_{band}')

            constraints += [
                gamma_prev >= 0,
                gamma_next >= 0,
                gamma_prev <= alpha_prev,
                gamma_next <= alpha_next,
                cp.diff(gamma_prev) <= cp.diff(alpha_prev),
                cp.diff(gamma_next) >= cp.diff(alpha_next),
            ]

            Sband_dj = S_dj[self.bins[band]]
            Sband_prev = S_prev[self.bins[band]]
            Sband_next = S_next[self.bins[band]]

            alpha_prev_ = cp.reshape(alpha_prev, shape=(1, num_frames))
            alpha_next_ = cp.reshape(alpha_next, shape=(1, num_frames))
            gamma_prev_ = cp.reshape(gamma_prev, shape=(1, num_frames))
            gamma_next_ = cp.reshape(gamma_next, shape=(1, num_frames))

            H_min = self.band_gains[band].reshape(-1, 1)
            H_inv = 1 - H_min

            # Precompute constant matrices (numpy) so variables appear only linearly (pure LP)
            A_prev = H_min * Sband_prev   # (n_bins_band, num_frames), numpy
            B_prev = H_inv * Sband_prev
            A_next = H_min * Sband_next
            B_next = H_inv * Sband_next

            Y_prev = cp.multiply(alpha_prev_, A_prev) + cp.multiply(gamma_prev_, B_prev)
            Y_next = cp.multiply(alpha_next_, A_next) + cp.multiply(gamma_next_, B_next)
            Y = Y_prev + Y_next
            Y_true = Sband_dj

            band_mean = max(Sband_dj.mean(), 1e-10)
            loss = cp.sum(cp.abs(Y - Y_true)) / np.prod(Sband_dj.shape) / band_mean
            losses[band] = loss

        loss = cp.sum(list(losses.values()))
        objective = cp.Minimize(loss)
        prob = cp.Problem(objective, constraints)

        # ECOS only: fast LP solver, optimal_inaccurate accepted (verified sufficient).
        # SCS removed — it hangs on large transitions with no reliable time cap.
        for solver in ('ECOS',):
            try:
                prob.solve(solver=solver, verbose=verbose, max_iters=200)
                if prob.status in ('optimal', 'optimal_inaccurate'):
                    break
                log.warning(f"{solver} returned status {prob.status}, trying next solver")
            except cp.SolverError as e:
                log.warning(f"{solver} solver failed ({e}), trying next solver")

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            log.warning(f"Optimization status: {prob.status}")
            return None

        results = {}
        for deck in ['prev', 'next']:
            alpha = prob.var_dict[f'alpha_{deck}'].value
            results[f'fader_{deck}'] = alpha
            for band in ['low', 'mid', 'high']:
                gamma = prob.var_dict[f'gamma_{deck}_{band}'].value
                beta = gamma / (alpha + 1e-8)
                results[f'eq_{deck}_{band}'] = beta

        return results


def compute_spectrogram(audio, opt_config=None):
    """Compute amplitude mel spectrogram for optimization.

    Args:
        audio: 1D numpy array (mono audio at OPT_SR)
        opt_config: OptConfig instance

    Returns:
        (n_mels, n_frames) amplitude spectrogram
    """
    c = opt_config or OptConfig()
    S = librosa.feature.melspectrogram(
        y=audio, sr=c.sr, n_fft=c.n_fft, hop_length=c.hop,
        n_mels=c.num_mel_bins, fmin=c.fmin, fmax=c.fmax, power=1,
    )
    return S
