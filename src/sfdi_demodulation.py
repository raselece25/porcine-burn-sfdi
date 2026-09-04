"""
sfdi_demodulation.py

Three-phase demodulation for Spatial Frequency Domain Imaging (SFDI).

Given three images of a sample illuminated with a sinusoidal intensity
pattern at the same spatial frequency but shifted in phase by 0, 120,
and 240 degrees, this recovers the AC (modulation) amplitude and DC
(planar) amplitude at every pixel:

    AC = (sqrt(2) / 3) * sqrt((I1 - I2)^2 + (I2 - I3)^2 + (I3 - I1)^2)
    DC = (I1 + I2 + I3) / 3

This is the standard three-phase algorithm used in SFDI (Cuccia et al.,
"Modulated imaging: quantitative analysis and tomography of turbid
media in the spatial-frequency domain," Opt. Lett. 30(11), 2005).

Reference: D. J. Cuccia, F. Bevilacqua, A. J. Durkin, B. J. Tromberg,
"Modulated imaging: quantitative analysis and tomography of turbid
media in the spatial-frequency domain," Opt. Lett. 30, 1354-1356 (2005).
"""

from __future__ import annotations

import numpy as np


def demodulate_three_phase(i1: np.ndarray, i2: np.ndarray, i3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover AC and DC reflectance components from three phase-shifted images.

    Parameters
    ----------
    i1, i2, i3 : np.ndarray
        Raw camera images captured at the same spatial frequency with
        illumination phase-shifted by 0, 120, and 240 degrees respectively.
        Any shape is accepted as long as all three arrays match.

    Returns
    -------
    ac : np.ndarray
        Demodulated AC (modulated) amplitude, same shape as inputs.
    dc : np.ndarray
        Demodulated DC (planar) amplitude, same shape as inputs.
    """
    i1 = np.asarray(i1, dtype=float)
    i2 = np.asarray(i2, dtype=float)
    i3 = np.asarray(i3, dtype=float)
    if not (i1.shape == i2.shape == i3.shape):
        raise ValueError("i1, i2, i3 must have identical shapes")

    ac = (np.sqrt(2.0) / 3.0) * np.sqrt(
        (i1 - i2) ** 2 + (i2 - i3) ** 2 + (i3 - i1) ** 2
    )
    dc = (i1 + i2 + i3) / 3.0
    return ac, dc


def calibrate_reflectance(
    ac_sample: np.ndarray,
    dc_sample: np.ndarray,
    ac_ref: np.ndarray,
    dc_ref: np.ndarray,
    ref_reflectance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw demodulated amplitudes into calibrated diffuse reflectance.

    Uses a reference phantom of known diffuse reflectance (``ref_reflectance``,
    typically measured independently or supplied by the phantom manufacturer)
    imaged under identical illumination/camera settings to remove the
    instrument response and illumination non-uniformity.

    Rd_sample(fx) = (AC_sample / AC_ref) * ref_reflectance   [AC / modulated channel]
    Rd_sample(0)  = (DC_sample / DC_ref) * ref_reflectance   [DC / planar channel]

    Returns
    -------
    rd_ac, rd_dc : np.ndarray
        Calibrated diffuse reflectance at the modulation frequency and at
        fx = 0, respectively.
    """
    rd_ac = (ac_sample / np.clip(ac_ref, 1e-9, None)) * ref_reflectance
    rd_dc = (dc_sample / np.clip(dc_ref, 1e-9, None)) * ref_reflectance
    return rd_ac, rd_dc
