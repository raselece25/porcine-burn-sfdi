"""
diffusion_model.py

Forward and inverse diffusion-approximation model for Spatial Frequency
Domain Imaging (SFDI), used to relate tissue optical properties
(mu_a, mu_s') to spatial-frequency-dependent diffuse reflectance R_d(fx).

This implements the standard photon-diffusion forward model with an
extrapolated boundary condition described in:

    D. J. Cuccia, F. Bevilacqua, A. J. Durkin, S. Merritt, B. J. Tromberg,
    "Quantitation and mapping of tissue optical properties using modulated
    imaging," J. Biomed. Opt. 14(2), 024012 (2009).

It is a simplified, single-layer, semi-infinite-medium model intended for
demonstration and teaching. The multi-layer / laparoscopic forward model
used in the underlying laboratory work (BOIL Lab, Stony Brook University)
is not reproduced here -- see the associated publications for the full
multi-layer treatment.
"""

from __future__ import annotations

import numpy as np


def internal_reflectance_parameter(n_tissue: float = 1.4) -> float:
    """Return the boundary-condition parameter A for a tissue/air interface.

    Uses the Groenhuis approximation for the fraction of diffusely
    reflected light at the tissue-air boundary (n_air = 1.0):

        r_d ~= -1.440 * n^-2 + 0.710 * n^-1 + 0.668 + 0.0636 * n
        A = (1 + r_d) / (1 - r_d)
    """
    n = n_tissue
    r_d = -1.440 * n**-2 + 0.710 * n**-1 + 0.668 + 0.0636 * n
    return (1.0 + r_d) / (1.0 - r_d)


def diffuse_reflectance(mua: np.ndarray, musp: np.ndarray, fx: float, n_tissue: float = 1.4) -> np.ndarray:
    """Diffusion-approximation diffuse reflectance R_d at spatial frequency fx.

    Parameters
    ----------
    mua, musp : array_like [mm^-1]
        Absorption and reduced scattering coefficients. Broadcastable.
    fx : float [mm^-1]
        Spatial frequency of the illumination pattern (0 for planar / DC).
    n_tissue : float
        Tissue refractive index, used for the internal-reflectance
        boundary condition.

    Returns
    -------
    rd : np.ndarray
        Predicted diffuse reflectance, dimensionless (0-1).
    """
    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)

    mu_tr = mua + musp
    a_prime = musp / mu_tr  # reduced scattering albedo

    A = internal_reflectance_parameter(n_tissue)

    # Spatial-frequency-dependent effective attenuation coefficient.
    # At fx = 0 this reduces to the familiar mu_eff = sqrt(3 * mua * mu_tr).
    mu_eff_fx = np.sqrt(3.0 * mua * mu_tr + (2.0 * np.pi * fx) ** 2)

    ratio = mu_eff_fx / mu_tr
    rd = (3.0 * A * a_prime) / ((ratio + 1.0) * (ratio + 3.0 * A))
    return rd


def build_forward_lut(
    mua_grid: np.ndarray,
    musp_grid: np.ndarray,
    fx_list: tuple[float, ...],
    n_tissue: float = 1.4,
) -> dict[float, np.ndarray]:
    """Precompute a forward lookup table of R_d(mua, musp) for each fx.

    Returns a dict mapping each spatial frequency to a 2-D array of shape
    (len(mua_grid), len(musp_grid)) with the predicted diffuse reflectance.
    """
    mua_mesh, musp_mesh = np.meshgrid(mua_grid, musp_grid, indexing="ij")
    lut = {}
    for fx in fx_list:
        lut[fx] = diffuse_reflectance(mua_mesh, musp_mesh, fx, n_tissue=n_tissue)
    return lut


def invert_optical_properties(
    rd_measured: dict[float, np.ndarray],
    mua_grid: np.ndarray,
    musp_grid: np.ndarray,
    n_tissue: float = 1.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert measured R_d at >=2 spatial frequencies to recover mua, musp.

    For every pixel, searches the precomputed forward LUT (built on the fly
    from mua_grid x musp_grid) for the (mua, musp) pair whose predicted
    reflectance at each supplied spatial frequency best matches the
    measurement (least-squares over the supplied frequencies).

    Parameters
    ----------
    rd_measured : dict[float, np.ndarray]
        Mapping from spatial frequency -> measured (calibrated) diffuse
        reflectance image, all images the same shape.
    mua_grid, musp_grid : np.ndarray
        1-D coordinate grids defining the search-space resolution.

    Returns
    -------
    mua_map, musp_map : np.ndarray
        Recovered optical property maps, same shape as the input images.
    """
    fx_list = tuple(rd_measured.keys())
    sample_shape = next(iter(rd_measured.values())).shape
    lut = build_forward_lut(mua_grid, musp_grid, fx_list, n_tissue=n_tissue)

    # Stack the LUT and measurements along a frequency axis for vectorized
    # least-squares matching: lut_stack has shape (n_fx, n_mua, n_musp).
    lut_stack = np.stack([lut[fx] for fx in fx_list], axis=0)
    meas_stack = np.stack([rd_measured[fx] for fx in fx_list], axis=0)
    n_pixels = int(np.prod(sample_shape))
    meas_flat = meas_stack.reshape(len(fx_list), n_pixels)

    mua_map = np.empty(n_pixels, dtype=float)
    musp_map = np.empty(n_pixels, dtype=float)

    lut_flat = lut_stack.reshape(len(fx_list), -1)  # (n_fx, n_mua*n_musp)
    mua_flat_grid = np.repeat(mua_grid, len(musp_grid))
    musp_flat_grid = np.tile(musp_grid, len(mua_grid))

    for p in range(n_pixels):
        target = meas_flat[:, p][:, None]  # (n_fx, 1)
        sq_err = np.sum((lut_flat - target) ** 2, axis=0)  # (n_mua*n_musp,)
        best = int(np.argmin(sq_err))
        mua_map[p] = mua_flat_grid[best]
        musp_map[p] = musp_flat_grid[best]

    return mua_map.reshape(sample_shape), musp_map.reshape(sample_shape)
