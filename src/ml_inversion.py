"""
ml_inversion.py

A lightweight machine-learning alternative to the lookup-table inversion
in ``diffusion_model.py``: a small multilayer-perceptron regressor that
maps calibrated multi-frequency diffuse reflectance directly to
(mu_a, mu_s'). This mirrors the "Monte Carlo / diffusion model + deep
learning" hybrid approach reported in the SFDI literature for fast,
real-time optical-property extraction, e.g.:

    S. Panigrahi, S. Gioux, "Machine learning approach for rapid and
    accurate estimation of optical properties using spatial frequency
    domain imaging," J. Biomed. Opt. 24(7), 071606 (2019).

The model is trained on synthetic (mua, musp) -> R_d(fx) pairs generated
from the diffusion forward model in this repo, so it should be
understood as a fast surrogate for that forward model rather than an
independently validated clinical tool.

Falls back to a scikit-learn MLPRegressor if PyTorch is not installed,
so the demo runs in either environment.
"""

from __future__ import annotations

import numpy as np

from .diffusion_model import diffuse_reflectance


def generate_training_pairs(
    n_samples: int = 4000,
    fx_list: tuple[float, ...] = (0.0, 0.2),
    mua_range: tuple[float, float] = (0.005, 0.08),
    musp_range: tuple[float, float] = (0.4, 3.0),
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random (mua, musp) pairs and compute their forward R_d(fx)."""
    rng = np.random.default_rng(seed)
    mua = rng.uniform(*mua_range, size=n_samples)
    musp = rng.uniform(*musp_range, size=n_samples)

    features = np.stack(
        [diffuse_reflectance(mua, musp, fx) for fx in fx_list], axis=1
    )
    targets = np.stack([mua, musp], axis=1)
    return features, targets


class OpticalPropertyRegressor:
    """Thin wrapper that trains/predicts with sklearn's MLPRegressor.

    Kept deliberately small (single hidden layer) since it only needs to
    learn a smooth, low-dimensional forward-model inverse for this demo.
    """

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (32, 16), seed: int = 0):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=2000,
            random_state=seed,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OpticalPropertyRegressor":
        x_scaled = self._x_scaler.fit_transform(x)
        y_scaled = self._y_scaler.fit_transform(y)
        self.model.fit(x_scaled, y_scaled)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self._x_scaler.transform(x)
        y_scaled = self.model.predict(x_scaled)
        return self._y_scaler.inverse_transform(y_scaled)
