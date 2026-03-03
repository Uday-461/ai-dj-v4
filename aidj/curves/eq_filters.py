"""
Audio EQ biquad filter design from cookbook formulae.

Ported from djmix-dataset:
  - djmix/eq/biquad_filters.py (shelf, peaking, _transform)
  - djmix/eq/eq3.py (eq3_filters, bin_gains)

Based on: https://gist.github.com/endolith/5455375

Python/SciPy implementation of the filters described in
"Cookbook formulae for audio EQ biquad filter coefficients"
by Robert Bristow-Johnson
https://www.musicdsp.org/en/latest/Filters/197-rbj-audio-eq-cookbook.html

These functions will output analog or digital transfer functions, deriving
the latter using the bilinear transform, as is done in the reference.
"""

from math import pi, tan, sinh
from math import log as ln
from cmath import sqrt

import numpy as np
from scipy.signal import tf2zpk, tf2ss, tf2sos, lp2lp, bilinear, sosfreqz


def _transform(b, a, Wn, analog, output):
    """Convert analog prototype filter to desired output format.

    Shift prototype filter to desired frequency, convert to digital with
    pre-warping, and return in various formats.
    """
    Wn = np.asarray(Wn)
    if not analog:
        if np.any(Wn < 0) or np.any(Wn > 1):
            raise ValueError(
                "Digital filter critical frequencies must be 0 <= Wn <= 1"
            )
        fs = 2.0
        warped = 2 * fs * tan(pi * Wn / fs)
    else:
        warped = Wn

    # Shift frequency
    b, a = lp2lp(b, a, wo=warped)

    # Find discrete equivalent if necessary
    if not analog:
        b, a = bilinear(b, a, fs=fs)

    # Transform to proper out type (pole-zero, numer-denom, state-space)
    if output in ('zpk', 'zp'):
        return tf2zpk(b, a)
    elif output in ('ba', 'tf'):
        return b, a
    elif output in ('ss', 'abcd'):
        return tf2ss(b, a)
    elif output in ('sos',):
        return tf2sos(b, a)
    else:
        raise ValueError('Unknown output type {0}'.format(output))


def shelf(Wn, dBgain, S=1, btype='low', ftype='half', analog=False,
          output='ba'):
    """Design an analog or digital biquad shelving filter with variable slope.

    Parameters
    ----------
    Wn : float
        Turnover frequency of the filter, defined by the ``ftype`` parameter.
        For digital filters, ``Wn`` is normalized from 0 to 1, where 1 is the
        Nyquist frequency.
        For analog filters, ``Wn`` is an angular frequency (e.g. rad/s).
    dBgain : float
        The gain at the shelf, in dB. Positive for boost, negative for cut.
    S : float, optional
        Shelf slope parameter. When S = 1, the shelf slope is as steep as it
        can be and remain monotonically increasing or decreasing gain.
    btype : {'low', 'high'}, optional
        Band type of the filter, low shelf or high shelf.
    ftype : {'half', 'outer', 'inner'}, optional
        Definition of the filter's turnover frequency.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'ss', 'sos'}, optional
        Type of output. Default is 'ba'.

    Returns
    -------
    Depends on ``output``.
    """
    Q = None

    if ftype in ('mid', 'half'):
        A = 10.0 ** (dBgain / 40.0)

        if Q is None:
            Q = 1 / sqrt((A + 1 / A) * (1 / S - 1) + 2)

        Az = A
        Ap = A

    elif ftype in ('outer',):
        A = 10.0 ** (dBgain / 20.0)

        if Q is None:
            Q = 1 / sqrt((A + 1 / A) * (1 / S - 1) + 2)

        if dBgain > 0:  # boost
            Az = A
            Ap = 1
        else:  # cut
            Az = 1
            Ap = A

    elif ftype in ('inner',):
        A = 10.0 ** (dBgain / 20.0)

        if Q is None:
            Q = 1 / sqrt((A + 1 / A) * (1 / S - 1) + 2)

        if dBgain > 0:  # boost
            Az = 1
            Ap = A
        else:  # cut
            Az = A
            Ap = 1
    else:
        raise ValueError('"%s" is not a known shelf type' % ftype)

    if btype == 'low':
        # H(s) = A * (s**2 + (sqrt(A)/Q)*s + A) / (A*s**2 + (sqrt(A)/Q)*s + 1)
        b = Ap * np.array([1, sqrt(Az) / Q, Az])
        a = np.array([Ap, sqrt(Ap) / Q, 1])
    elif btype == 'high':
        # H(s) = A * (A*s**2 + (sqrt(A)/Q)*s + 1) / (s**2 + (sqrt(A)/Q)*s + A)
        b = Ap * np.array([Az, sqrt(Az) / Q, 1])
        a = np.array([1, sqrt(Ap) / Q, Ap])
    else:
        raise ValueError('"%s" is not a known shelf type' % btype)

    return _transform(b, a, Wn, analog, output)


def peaking(Wn, dBgain, Q=None, BW=None, type='half', analog=False,
            output='ba'):
    """Design an analog or digital biquad peaking EQ filter with variable Q.

    Transfer function: H(s) = (s**2 + s*(Az/Q) + 1) / (s**2 + s/(Ap*Q) + 1)

    Used in graphic or parametric EQs.

    Parameters
    ----------
    Wn : float
        Center frequency of the filter.
        For digital filters, ``Wn`` is normalized from 0 to 1, where 1 is the
        Nyquist frequency.
        For analog filters, ``Wn`` is an angular frequency (e.g. rad/s).
    dBgain : float
        The gain at the center frequency, in dB.
    Q : float, optional
        Quality factor of the filter.
    BW : float, optional
        Bandwidth in octaves. Used if Q is None.
    type : {'half', 'constantq'}, optional
        Where on the curve to measure the bandwidth.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter.
    output : {'ba', 'zpk', 'ss', 'sos'}, optional
        Type of output. Default is 'ba'.

    Returns
    -------
    Depends on ``output``.
    """
    if Q is None and BW is None:
        BW = 1  # octave

    if Q is None:
        Q = 1 / (2 * sinh(ln(2) / 2 * BW))  # analog filter prototype

    if type in ('half',):
        A = 10.0 ** (dBgain / 40.0)
        Az = A
        Ap = A
    elif type in ('constantq',):
        A = 10.0 ** (dBgain / 20.0)
        if dBgain > 0:  # boost
            Az = A
            Ap = 1
        else:  # cut
            Az = 1
            Ap = A
    else:
        raise ValueError('"%s" is not a known peaking type' % type)

    # H(s) = (s**2 + s*(Az/Q) + 1) / (s**2 + s/(Ap*Q) + 1)
    b = np.array([1, Az / Q, 1])
    a = np.array([1, 1 / (Ap * Q), 1])

    return _transform(b, a, Wn, analog, output)


def eq3_filters(
    cutoff_low,
    center_mid,
    cutoff_high,
    sr,
    low_db_gain=-80,
    mid_db_gain=-27,
    high_db_gain=-80,
    mid_Q=3,
):
    """Create a 3-band EQ filter bank (low shelf, mid peak, high shelf).

    Returns SOS (second-order sections) representations for each band filter
    at its maximum attenuation setting.

    Parameters
    ----------
    cutoff_low : float
        Low shelf cutoff frequency in Hz.
    center_mid : float
        Mid-band center frequency in Hz.
    cutoff_high : float
        High shelf cutoff frequency in Hz.
    sr : int
        Sample rate in Hz.
    low_db_gain : float
        Low shelf gain in dB (typically a large negative number).
    mid_db_gain : float
        Mid peak gain in dB (typically a large negative number).
    high_db_gain : float
        High shelf gain in dB (typically a large negative number).
    mid_Q : float
        Quality factor for the mid-band peaking filter.

    Returns
    -------
    sos_low, sos_mid, sos_high : ndarray
        SOS filter representations for each band.
    """
    nyq = 0.5 * sr
    sos_low = shelf(
        cutoff_low / nyq, dBgain=low_db_gain,
        btype='low', ftype='inner', output='sos',
    )
    sos_mid = peaking(
        center_mid / nyq, dBgain=mid_db_gain,
        Q=mid_Q, type='constantq', output='sos',
    )
    sos_high = shelf(
        cutoff_high / nyq, dBgain=high_db_gain,
        btype='high', ftype='inner', output='sos',
    )

    return sos_low, sos_mid, sos_high


def bin_gains(filter_sos, bin_frequencies, sr):
    """Compute the gain of a filter at specific frequency bins.

    Parameters
    ----------
    filter_sos : ndarray
        Filter in second-order sections format.
    bin_frequencies : ndarray
        Array of frequencies (in Hz) at which to evaluate the filter gain.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    ndarray
        Absolute gain at each bin frequency.
    """
    w, h = sosfreqz(filter_sos, worN=8192)
    filt_freqs = (sr * 0.5 / np.pi) * w
    filt_gains = np.abs(h)

    # Find the closest filter frequency for each bin frequency.
    dist = np.abs(filt_freqs - bin_frequencies.reshape(-1, 1))
    i_closest = dist.argmin(axis=1)
    result = filt_gains[i_closest]

    return result
