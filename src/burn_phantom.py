"""
burn_phantom.py

Synthetic burn-severity phantom generator.

IMPORTANT: This module generates *synthetic* optical-property maps that
are only loosely inspired by the burn-injury optics literature (thermal
coagulation of dermal collagen increases reduced scattering mu_s' and,
at greater depth/severity, increases absorption mu_a as blood pools and
denatures). It does NOT contain, reproduce, or derive from any
unpublished experimental data collected in the BOIL Lab porcine burn
study. It exists purely to give the rest of this repository (SFDI
demodulation, diffusion-model inversion, Monte Carlo validation, and the
ML inversion demo) realistic-looking, reproducible test data to run on.

For the real, IACUC-approved porcine burn study results, see:
Kluiszo, Belcastro, Ahmmed et al., "Burn Monitoring Using SFDI in a
Porcine Model," Proc. SPIE 13823, 49-52 (2026).
"""

from __future__ import annotations

import numpy as np

# Baseline (unburned, healthy skin) optical properties at a representative
# visible/NIR SFDI wavelength (~660 nm), broadly consistent with published
# in-vivo skin ranges (Cuccia et al. 2009; Nguyen et al. 2013).
BASELINE_MUA = 0.015   # mm^-1
BASELINE_MUSP = 1.0    # mm^-1

# Burn severity is parameterized on a continuous 0 (unburned) -> 1
# (full-thickness) scale. These synthetic coefficients set how much mua and
# musp shift with severity; they are illustrative, not calibrated.
MUA_SEVERITY_GAIN = 0.06     # mm^-1 at severity = 1
MUSP_SEVERITY_GAIN = 1.8     # mm^-1 at severity = 1


def make_severity_map(size: int = 32, seed: int | None = 0) -> np.ndarray:
    """Create a synthetic 2-D burn-severity map (values in [0, 1]).

    Uses a handful of overlapping radial-gradient "burn zones" so the
    result resembles a burn with a hotter core and a paler periphery,
    which is a common qualitative pattern in scald/contact burns.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    severity = np.zeros((size, size), dtype=float)

    n_zones = 2
    for _ in range(n_zones):
        cx, cy = rng.uniform(0.3, 0.7, size=2) * size
        radius = rng.uniform(0.15, 0.3) * size
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        zone = np.clip(1.0 - dist / radius, 0.0, 1.0) ** 1.5
        severity = np.maximum(severity, zone)

    severity += rng.normal(0.0, 0.02, size=severity.shape)
    return np.clip(severity, 0.0, 1.0)


def severity_to_optical_properties(severity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a severity field to synthetic (mua, musp) fields."""
    mua = BASELINE_MUA + MUA_SEVERITY_GAIN * severity
    musp = BASELINE_MUSP + MUSP_SEVERITY_GAIN * severity
    return mua, musp
