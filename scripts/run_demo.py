#!/usr/bin/env python3
"""
run_demo.py

End-to-end demonstration of the repository's SFDI burn-severity pipeline
on synthetic data:

  1. Generate a synthetic burn-severity phantom and its (mua, musp) maps.
  2. Forward-simulate calibrated multi-frequency diffuse reflectance
     using the diffusion approximation.
  3. Recover (mua, musp) two ways: (a) lookup-table inversion, and
     (b) a trained ML regressor.
  4. Cross-check the diffusion model against the Monte Carlo simulator
     at a few sample optical-property combinations (fx = 0 only).
  5. Save a summary figure to figures/demo_summary.png.

Run with:  python -m scripts.run_demo   (from the repo root)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.burn_phantom import make_severity_map, severity_to_optical_properties
from src.diffusion_model import diffuse_reflectance, invert_optical_properties
from src.ml_inversion import OpticalPropertyRegressor, generate_training_pairs
from src.monte_carlo import simulate_diffuse_reflectance

FX_LIST = (0.0, 0.2)  # mm^-1


def main() -> None:
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("[1/5] Generating synthetic burn-severity phantom...")
    severity = make_severity_map(size=24, seed=1)
    mua_true, musp_true = severity_to_optical_properties(severity)

    print("[2/5] Forward-simulating multi-frequency diffuse reflectance...")
    rd_measured = {fx: diffuse_reflectance(mua_true, musp_true, fx) for fx in FX_LIST}
    # Add a small amount of noise to emulate camera/shot noise.
    rng = np.random.default_rng(0)
    for fx in FX_LIST:
        rd_measured[fx] = np.clip(
            rd_measured[fx] + rng.normal(0.0, 0.002, size=rd_measured[fx].shape), 0.0, 1.0
        )

    print("[3/5] Recovering optical properties via lookup-table inversion...")
    t0 = time.time()
    mua_grid = np.linspace(0.005, 0.08, 60)
    musp_grid = np.linspace(0.4, 3.0, 60)
    mua_lut, musp_lut = invert_optical_properties(rd_measured, mua_grid, musp_grid)
    print(f"      done in {time.time() - t0:.2f} s")

    print("[3b/5] Training ML regressor and recovering via ML inversion...")
    x_train, y_train = generate_training_pairs(n_samples=6000, fx_list=FX_LIST, seed=0)
    regressor = OpticalPropertyRegressor().fit(x_train, y_train)
    x_query = np.stack([rd_measured[fx].ravel() for fx in FX_LIST], axis=1)
    y_pred = regressor.predict(x_query)
    mua_ml = y_pred[:, 0].reshape(severity.shape)
    musp_ml = y_pred[:, 1].reshape(severity.shape)

    mua_rmse_lut = float(np.sqrt(np.mean((mua_lut - mua_true) ** 2)))
    mua_rmse_ml = float(np.sqrt(np.mean((mua_ml - mua_true) ** 2)))
    print(f"      mu_a RMSE  -- LUT: {mua_rmse_lut:.4f} mm^-1   ML: {mua_rmse_ml:.4f} mm^-1")

    print("[4/5] Comparing diffusion model vs. Monte Carlo at fx=0 (qualitative check)...")
    print("      NOTE: the diffusion model is a first-order analytic approximation and")
    print("      the Monte Carlo simulator here is a coarse, teaching-scale implementation;")
    print("      they are expected to agree in trend (Rd falls as mu_a rises) but not to")
    print("      sub-percent precision. See README.md for scope/limitations.")
    test_points = [(0.01, 1.0), (0.02, 1.5), (0.05, 2.2)]
    print("      mu_a    mu_s'   R_d(diffusion)   R_d(Monte Carlo, N=20000)")
    for mua_t, musp_t in test_points:
        rd_diff = float(diffuse_reflectance(np.array(mua_t), np.array(musp_t), 0.0))
        rd_mc = simulate_diffuse_reflectance(mua_t, musp_t, n_photons=20_000, seed=1)
        print(f"      {mua_t:.3f}   {musp_t:.2f}    {rd_diff:.4f}           {rd_mc:.4f}")

    print("[5/5] Saving summary figure...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    im0 = axes[0, 0].imshow(severity, cmap="inferno", vmin=0, vmax=1)
    axes[0, 0].set_title("Synthetic burn severity (ground truth)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(mua_true, cmap="viridis")
    axes[0, 1].set_title("mu_a, true [mm^-1]")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(musp_true, cmap="viridis")
    axes[0, 2].set_title("mu_s', true [mm^-1]")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(mua_lut, cmap="viridis")
    axes[1, 0].set_title(f"mu_a, LUT-recovered\nRMSE={mua_rmse_lut:.4f}")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(mua_ml, cmap="viridis")
    axes[1, 1].set_title(f"mu_a, ML-recovered\nRMSE={mua_rmse_ml:.4f}")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    im5 = axes[1, 2].imshow(musp_lut, cmap="viridis")
    axes[1, 2].set_title("mu_s', LUT-recovered")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("SFDI burn-severity demo pipeline (synthetic data)", fontsize=13)
    fig.tight_layout()
    out_path = figures_dir / "demo_summary.png"
    fig.savefig(out_path, dpi=150)
    print(f"      saved to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
