import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_model import diffuse_reflectance, invert_optical_properties
from src.monte_carlo import simulate_diffuse_reflectance
from src.sfdi_demodulation import calibrate_reflectance, demodulate_three_phase


def test_demodulate_three_phase_recovers_known_ac_dc():
    ac_true, dc_true = 0.3, 0.5
    phases = np.deg2rad([0, 120, 240])
    images = [dc_true + ac_true * np.cos(p) for p in phases]

    ac, dc = demodulate_three_phase(*images)

    assert np.isclose(ac, ac_true, atol=1e-6)
    assert np.isclose(dc, dc_true, atol=1e-6)


def test_calibrate_reflectance_scales_by_reference():
    ac_sample = np.array([2.0])
    dc_sample = np.array([4.0])
    ac_ref = np.array([1.0])
    dc_ref = np.array([2.0])

    rd_ac, rd_dc = calibrate_reflectance(ac_sample, dc_sample, ac_ref, dc_ref, ref_reflectance=0.5)

    assert np.isclose(rd_ac[0], 1.0)
    assert np.isclose(rd_dc[0], 1.0)


def test_diffuse_reflectance_decreases_with_absorption():
    musp = np.array([1.0, 1.0])
    mua = np.array([0.01, 0.05])
    rd = diffuse_reflectance(mua, musp, fx=0.0)
    assert rd[0] > rd[1]  # more absorption -> less reflectance


def test_invert_optical_properties_recovers_known_values_on_grid():
    mua_true = np.array([[0.02]])
    musp_true = np.array([[1.5]])
    fx_list = (0.0, 0.2)
    rd_measured = {fx: diffuse_reflectance(mua_true, musp_true, fx) for fx in fx_list}

    mua_grid = np.linspace(0.005, 0.08, 40)
    musp_grid = np.linspace(0.4, 3.0, 40)
    mua_hat, musp_hat = invert_optical_properties(rd_measured, mua_grid, musp_grid)

    assert np.isclose(mua_hat[0, 0], 0.02, atol=0.01)
    assert np.isclose(musp_hat[0, 0], 1.5, atol=0.15)


def test_monte_carlo_and_diffusion_model_agree_qualitatively():
    """The Monte Carlo simulator is a coarse, teaching-scale implementation
    and the diffusion model is a first-order analytic approximation, so we
    do not expect tight (sub-percent) quantitative agreement between them.
    This test instead checks the qualitative behavior both models must
    share: diffuse reflectance decreases monotonically as absorption
    increases, and both stay within a physically valid [0, 1] range.
    See README.md for a discussion of this repo's validation scope.
    """
    musp = 1.5
    mua_values = [0.005, 0.02, 0.05, 0.1]

    rd_diffusion = [
        float(diffuse_reflectance(np.array(mua), np.array(musp), fx=0.0)) for mua in mua_values
    ]
    rd_mc = [
        simulate_diffuse_reflectance(mua, musp, n_photons=8_000, seed=42) for mua in mua_values
    ]

    for rd in rd_diffusion + rd_mc:
        assert 0.0 <= rd <= 1.0

    assert all(a > b for a, b in zip(rd_diffusion, rd_diffusion[1:]))
    assert all(a > b for a, b in zip(rd_mc, rd_mc[1:]))
