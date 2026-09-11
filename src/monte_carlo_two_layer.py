"""
monte_carlo_two_layer.py

Standalone two-layer Monte Carlo photon-transport simulator for a turbid
medium with a finite top layer over a semi-infinite bottom layer -- e.g.
epidermis over dermis. Self-contained: no import of / dependency on a
single-layer model.

Follows the standard photon-packet weighting scheme described in:

    L. Wang, S. L. Jacques, L. Zheng, "MCML -- Monte Carlo modeling of
    light transport in multi-layered tissues," Comput. Methods Programs
    Biomed. 47(2), 131-146 (1995).

Motivation (why two layers, not homogeneous): Poon, "Early Assessment of
Burn Severity in Human Tissue with Multi-Wavelength Spatial Frequency
Domain Imaging" (M.S. thesis, Wright State Univ., 2016), Sec. 4.5.1 and
Ch. 5, shows via a single-layer-vs-two-layer Monte Carlo comparison
(Wang et al.'s MC code) that fitting burned skin with a homogeneous
diffusion model can lose real physiological information, since even
plain skin is at least epidermis + dermis, and proposes resolving the
top layer's (mu_a1, mu_s1') with high spatial frequencies plus an
independent layer-thickness measurement (there, HFUS), then fitting the
bottom layer's (mu_a2, mu_s2') at low spatial frequencies. This module
is a from-scratch two-layer forward model in that same spirit, useful
as an independent MC check for that kind of two-layer SFDI fit.

Layer 1 occupies 0 <= z < thickness1; layer 2 occupies z >= thickness1
and is semi-infinite. Teaching-scale: total diffuse reflectance only, no
spatial-frequency modulation, no lateral photon-position tallying.

Boundary bookkeeping: a photon's step is drawn once as a dimensionless
number of mean free paths (`rem`, ~Exp(1)). By the memoryless property
of the exponential distribution this budget is valid regardless of
which layer's mu_t converts it to physical distance, so it carries
unchanged across a layer boundary. Each outer-loop iteration does ONE
of two things per photon: (a) the remaining budget is smaller than the
distance to the next boundary, so the photon reaches its next
scattering site inside the current layer -- absorption is deposited and
a new Henyey-Greenstein scattering direction is drawn, then a fresh
`rem` is drawn for the next hop; or (b) the boundary is reached first,
so the photon is moved exactly there, `rem` is decremented by the
distance already spent, and Fresnel reflection / Snell refraction
decides whether it reflects back into the same layer, transmits into
the other layer, or (at the top surface) escapes as diffuse
reflectance. No new `rem` is drawn in branch (b) -- the same hop keeps
resolving on later iterations, so `max_steps` bounds "scatters +
crossings" combined.

ponytail: a layer 1 much thinner than a mean free path needs more
boundary crossings to resolve each hop, which eats into the same
max_steps budget used for actual scattering events -- if convergence
looks off for a very thin/high-scattering layer 1, raise max_steps
first before suspecting the physics.
"""

from __future__ import annotations

import numpy as np


def simulate_diffuse_reflectance_two_layer(
    mua1: float,
    musp1: float,
    mua2: float,
    musp2: float,
    thickness1: float,
    g1: float = 0.8,
    g2: float = 0.8,
    n1: float = 1.4,
    n2: float = 1.4,
    n_ambient: float = 1.0,
    n_photons: int = 20_000,
    weight_threshold: float = 1e-4,
    roulette_m: int = 10,
    max_steps: int = 8000,
    seed: int | None = 0,
) -> float:
    """Estimate total diffuse reflectance for a two-layer turbid medium.

    Parameters
    ----------
    mua1, musp1, g1, n1 : layer-1 (top, e.g. epidermis) optical properties
        [mm^-1], anisotropy, refractive index.
    mua2, musp2, g2, n2 : layer-2 (semi-infinite, e.g. dermis) optical
        properties, same units.
    thickness1 : float [mm]
        Layer-1 thickness. Layer 2 extends from thickness1 to infinity.
    n_ambient : float
        Refractive index of the medium above the tissue (air).
    n_photons : int
        Number of photon packets to launch.
    weight_threshold, roulette_m : float, int
        Russian-roulette termination parameters.
    max_steps : int
        Hard cap on scatters + boundary crossings per photon (safety limit).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    rd : float
        Estimated total diffuse reflectance at fx = 0. Passing
        thickness1 = 0 (or making layer 2 identical to layer 1) should
        make this equivalent to a homogeneous-medium simulation with
        layer 1's properties -- see `demo()` below for a self-contained
        check of exactly that.
    """
    rng = np.random.default_rng(seed)
    mus1, mus2 = musp1 / (1.0 - g1), musp2 / (1.0 - g2)
    mu_t = np.array([mua1 + mus1, mua2 + mus2])
    mua_arr = np.array([mua1, mua2])
    g_arr = np.array([g1, g2])
    n_arr = np.array([n1, n2])

    r_sp = ((n1 - n_ambient) / (n1 + n_ambient)) ** 2

    n = n_photons
    weight = np.full(n, 1.0 - r_sp, dtype=float)
    pos = np.zeros((n, 3), dtype=float)
    direction = np.zeros((n, 3), dtype=float)
    direction[:, 2] = 1.0
    layer = np.zeros(n, dtype=np.int8)  # 0 = layer 1, 1 = layer 2

    alive = np.ones(n, dtype=bool)
    rem = -np.log(np.clip(rng.random(n), 1e-12, 1.0))  # dimensionless budget per photon
    reflected_weight = 0.0

    for _ in range(max_steps):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break

        cur_layer = layer[idx]
        uz = direction[idx, 2]
        z = pos[idx, 2]
        mt = mu_t[cur_layer]

        target = np.full(idx.size, np.nan)
        target[(cur_layer == 0) & (uz > 0)] = thickness1  # layer1 -> internal, going down
        target[(cur_layer == 0) & (uz < 0)] = 0.0          # layer1 -> top surface, going up
        target[(cur_layer == 1) & (uz < 0)] = thickness1   # layer2 -> internal, going up
        # (cur_layer == 1) & (uz >= 0): semi-infinite below, no boundary -> stays NaN.

        has_target = ~np.isnan(target)
        s_to_target = np.full(idx.size, np.inf)
        s_to_target[has_target] = np.clip(
            (target[has_target] - z[has_target]) * mt[has_target] / uz[has_target],
            0.0, None,
        )

        scatters_first = rem[idx] <= s_to_target
        scat_idx = idx[scatters_first]
        cross_idx = idx[~scatters_first]

        # --- Reaches next scattering site inside the current layer ---
        if scat_idx.size:
            dist = rem[scat_idx] / mu_t[layer[scat_idx]]
            pos[scat_idx] += direction[scat_idx] * dist[:, None]
            frac = mua_arr[layer[scat_idx]] / mu_t[layer[scat_idx]]
            weight[scat_idx] *= 1.0 - frac

            m = scat_idx.size
            rnd2 = rng.random(m)
            gsc = g_arr[layer[scat_idx]]
            cos_theta = np.where(
                np.abs(gsc) > 1e-3,
                (1.0 / (2.0 * gsc + 1e-30))
                * (1.0 + gsc**2 - ((1.0 - gsc**2) / (1.0 - gsc + 2.0 * gsc * rnd2)) ** 2),
                2.0 * rnd2 - 1.0,
            )
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            phi = 2.0 * np.pi * rng.random(m)
            direction[scat_idx] = _rotate_direction(direction[scat_idx], cos_theta, sin_theta, phi)

            low = scat_idx[weight[scat_idx] < weight_threshold]
            if low.size:
                survive = rng.random(low.size) < (1.0 / roulette_m)
                weight[low[survive]] *= roulette_m
                alive[low[~survive]] = False

            still = scat_idx[alive[scat_idx]]
            if still.size:
                rem[still] = -np.log(np.clip(rng.random(still.size), 1e-12, 1.0))

        # --- Hits a layer/surface boundary before its next scattering site ---
        if cross_idx.size:
            s_used = s_to_target[~scatters_first]
            dist = s_used / mu_t[layer[cross_idx]]
            pos[cross_idx] += direction[cross_idx] * dist[:, None]
            rem[cross_idx] -= s_used

            at_top = (layer[cross_idx] == 0) & (direction[cross_idx, 2] < 0)
            top_idx = cross_idx[at_top]
            int_idx = cross_idx[~at_top]

            if top_idx.size:
                cos_i = np.clip(-direction[top_idx, 2], 1e-6, 1.0)
                r_f = _fresnel_reflectance(cos_i, n1, n_ambient)
                transmitted = weight[top_idx] * (1.0 - r_f)
                reflected_weight += float(np.sum(transmitted))
                weight[top_idx] *= r_f
                pos[top_idx, 2] = -pos[top_idx, 2]
                direction[top_idx, 2] *= -1.0
                dead = top_idx[weight[top_idx] < weight_threshold * 1e-2]
                alive[dead] = False

            if int_idx.size:
                from_layer = layer[int_idx]
                to_layer = 1 - from_layer
                n_from, n_to = n_arr[from_layer], n_arr[to_layer]
                cos_i = np.clip(np.abs(direction[int_idx, 2]), 1e-6, 1.0)
                r_f = _fresnel_reflectance(cos_i, n_from, n_to)
                transmit = rng.random(int_idx.size) >= r_f

                reflect_idx = int_idx[~transmit]
                direction[reflect_idx, 2] *= -1.0

                transmit_idx = int_idx[transmit]
                if transmit_idx.size:
                    direction[transmit_idx] = _refract_direction(
                        direction[transmit_idx], n_from[transmit], n_to[transmit]
                    )
                    layer[transmit_idx] = to_layer[transmit]

    return reflected_weight / n_photons


def _fresnel_reflectance(cos_i: np.ndarray, n1: float, n2: float) -> np.ndarray:
    """Unpolarized Fresnel reflectance at an interface, with total internal
    reflection handled explicitly. (Inlined here so this module has no
    dependency on the single-layer model.)"""
    sin_i = np.sqrt(np.clip(1.0 - cos_i**2, 0.0, 1.0))
    sin_t = n1 / n2 * sin_i
    tir = sin_t >= 1.0
    sin_t_c = np.clip(sin_t, 0.0, 1.0 - 1e-12)
    cos_t = np.sqrt(1.0 - sin_t_c**2)

    rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
    r = 0.5 * (rs + rp)
    r = np.where(tir, 1.0, r)
    return r


def _rotate_direction(
    d: np.ndarray, cos_theta: np.ndarray, sin_theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """Rotate each direction vector in d by polar angle theta and azimuth
    phi (Henyey-Greenstein scattering step). Inlined -- no external
    dependency."""
    ux, uy, uz = d[:, 0], d[:, 1], d[:, 2]
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    denom = np.sqrt(np.clip(1.0 - uz**2, 1e-12, None))
    near_pole = np.abs(uz) > 0.99999

    new_x = np.where(
        near_pole,
        sin_theta * cos_phi,
        sin_theta * (ux * uz * cos_phi - uy * sin_phi) / denom + ux * cos_theta,
    )
    new_y = np.where(
        near_pole,
        sin_theta * sin_phi,
        sin_theta * (uy * uz * cos_phi + ux * sin_phi) / denom + uy * cos_theta,
    )
    new_z = np.where(
        near_pole,
        np.sign(uz) * cos_theta,
        -sin_theta * cos_phi * denom + uz * cos_theta,
    )

    out = np.stack([new_x, new_y, new_z], axis=1)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norm, 1e-12, None)


def _refract_direction(d: np.ndarray, n_from: np.ndarray, n_to: np.ndarray) -> np.ndarray:
    """Snell's-law refraction at a horizontal (z = const) layer boundary,
    preserving the photon's up/down travel sense. Only called on photons
    already selected as transmitting (not totally internally reflected)."""
    ux, uy, uz = d[:, 0], d[:, 1], d[:, 2]
    cos_i = np.clip(np.abs(uz), 1e-6, 1.0)
    sin_i = np.sqrt(np.clip(1.0 - cos_i**2, 0.0, 1.0))
    sin_t = np.clip(n_from / n_to * sin_i, 0.0, 1.0 - 1e-12)
    cos_t = np.sqrt(1.0 - sin_t**2)

    scale = np.where(sin_i > 1e-6, sin_t / np.clip(sin_i, 1e-6, None), 1.0)
    out = np.stack([ux * scale, uy * scale, np.sign(uz) * cos_t], axis=1)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norm, 1e-12, None)


def demo() -> None:
    """Self-contained sanity checks (no external model needed):

    1. thickness1 = 0 (layer 1 has zero thickness) must give the same Rd
       as a homogeneous medium with layer 2's properties, regardless of
       what layer 1's properties are set to, since no photon can ever
       occupy a zero-thickness layer.
    2. A two-layer medium where layer 2 is identical to layer 1 must
       match a "thick homogeneous layer1" run (thickness1 huge) within
       Monte Carlo noise -- both describe the same physical medium.
    3. Increasing mua1 (holding everything else fixed) must not increase
       Rd -- more absorption near the surface can only reduce or match
       the diffuse reflectance.
    4. Rd must always be a valid fraction in [0, 1].
    """
    mua, musp, g, n = 0.01, 1.0, 0.8, 1.4
    # ponytail: this low-mua regime needs most photons to run close to the
    # full max_steps budget before Russian roulette culls them (nothing to
    # do with layer count), so keep n_photons modest here or these checks
    # get slow -- bump it if you want tighter noise margins.
    n_photons = 20_000

    # (1) zero-thickness top layer <=> homogeneous layer 2.
    rd_zero_top = simulate_diffuse_reflectance_two_layer(
        mua1=0.5, musp1=5.0, mua2=mua, musp2=musp, thickness1=0.0,
        g1=g, g2=g, n1=n, n2=n, n_photons=n_photons, seed=1,
    )
    rd_homog2 = simulate_diffuse_reflectance_two_layer(
        mua1=mua, musp1=musp, mua2=mua, musp2=musp, thickness1=1e6,
        g1=g, g2=g, n1=n, n2=n, n_photons=n_photons, seed=1,
    )
    rel_diff = abs(rd_zero_top - rd_homog2) / rd_homog2
    print(f"zero-thickness top layer Rd = {rd_zero_top:.4f}, "
          f"homogeneous (layer-2 props) Rd = {rd_homog2:.4f}, rel diff = {rel_diff:.2%}")
    assert rel_diff < 0.15

    # (2) identical layers <=> thick homogeneous layer-1 medium.
    rd_two_identical = simulate_diffuse_reflectance_two_layer(
        mua, musp, mua, musp, thickness1=0.5, g1=g, g2=g, n1=n, n2=n,
        n_photons=n_photons, seed=2,
    )
    rd_thick_layer1 = simulate_diffuse_reflectance_two_layer(
        mua, musp, mua2=5.0, musp2=50.0, thickness1=50.0, g1=g, g2=g, n1=n, n2=n,
        n_photons=n_photons, seed=2,
    )
    rel_diff2 = abs(rd_two_identical - rd_thick_layer1) / rd_thick_layer1
    print(f"identical-layers Rd = {rd_two_identical:.4f}, "
          f"thick-layer1 (layer2 irrelevant) Rd = {rd_thick_layer1:.4f}, "
          f"rel diff = {rel_diff2:.2%}")
    assert rel_diff2 < 0.15

    # (3) monotonicity: more absorbing top layer -> lower or equal Rd.
    rd_low_mua1 = simulate_diffuse_reflectance_two_layer(
        mua1=0.005, musp1=musp, mua2=mua, musp2=musp, thickness1=0.2,
        g1=g, g2=g, n1=n, n2=n, n_photons=n_photons, seed=3,
    )
    rd_high_mua1 = simulate_diffuse_reflectance_two_layer(
        mua1=0.2, musp1=musp, mua2=mua, musp2=musp, thickness1=0.2,
        g1=g, g2=g, n1=n, n2=n, n_photons=n_photons, seed=3,
    )
    print(f"low mua1 Rd = {rd_low_mua1:.4f}, high mua1 Rd = {rd_high_mua1:.4f}")
    assert rd_high_mua1 < rd_low_mua1

    # (4) always a valid fraction.
    for rd in (rd_zero_top, rd_homog2, rd_two_identical, rd_thick_layer1,
               rd_low_mua1, rd_high_mua1):
        assert 0.0 <= rd <= 1.0


if __name__ == "__main__":
    demo()
