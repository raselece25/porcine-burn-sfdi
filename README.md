# SFDI Burn-Severity Optical Property Mapping

Python + MATLAB implementation of a Spatial Frequency Domain Imaging
(SFDI) pipeline for mapping tissue optical properties (absorption
coefficient `mu_a`, reduced scattering coefficient `mu_s'`) associated
with thermal burn injury, plus a diffusion-approximation forward model,
a Monte Carlo cross-check, and a machine-learning inversion alternative.

This repository accompanies ongoing research at the **BOIL Lab, Stony
Brook University** on SFDI-based characterization of skin's optical
response to thermal injury:

> Kluiszo, E.\*, Belcastro, L.\*, Ahmmed, R.\*, Singer, A., Clark, R., et al.
> (2026). "Burn Monitoring Using SFDI in a Porcine Model." *Photonics in
> Dermatology and Plastic Surgery*, Proc. SPIE 13823, 49-52.
> (\*equal contribution)

## What this is (and isn't)

This repo implements the **general computational methodology** —
three-phase SFDI demodulation, diffusion-approximation optical property
extraction, a Monte Carlo validation model, and a deep-learning-based
inversion shortcut — using **synthetic, randomly generated phantom
data**. It does **not** include any unpublished experimental data,
animal-study results, or the full multi-layer/laparoscopic forward model
used in the underlying IACUC-approved porcine study. The intent is to
give a clear, runnable, open reference implementation of the underlying
optics and signal-processing pipeline for portfolio and educational
purposes.

## Method overview

1. **Three-phase SFDI demodulation** (`src/sfdi_demodulation.py`,
   `matlab/sfdiDemodulate.m`): recovers AC (modulated) and DC (planar)
   reflectance from three phase-shifted structured-light images.
2. **Diffusion-approximation forward model** (`src/diffusion_model.py`,
   `matlab/diffuseReflectance.m`): predicts diffuse reflectance `R_d(fx)`
   from `(mu_a, mu_s')` at a given spatial frequency, following Cuccia et
   al., *J. Biomed. Opt.* 14(2), 024012 (2009).
3. **Lookup-table inversion** (`invert_optical_properties` /
   `invertOpticalProperties.m`): recovers per-pixel `(mu_a, mu_s')` from
   multi-frequency measurements by matching against a precomputed forward
   grid.
4. **Monte Carlo cross-check** (`src/monte_carlo.py`): an independent,
   vectorized single-layer Monte Carlo photon-transport simulator
   (Wang, Jacques & Zheng, *Comput. Methods Programs Biomed.* 47(2),
   131-146, 1995) used as a rough, qualitative cross-check against the
   diffusion model at `fx = 0`. **Scope note:** this is a coarse,
   teaching-scale MC (limited photon count, isotropic-scattering
   similarity approximation) compared against a first-order analytic
   diffusion model — the two are expected to agree in trend (reflectance
   falls as absorption rises) but not to tight quantitative precision.
   Neither model here has been validated against the peer-reviewed
   results in the citation above; treat both as illustrative rather than
   as a substitute for the published, experimentally-validated pipeline.
5. **ML-based inversion** (`src/ml_inversion.py`): a small MLP regressor
   trained on synthetic forward-model samples, offering a much faster
   (single forward pass) alternative to the lookup-table search, in the
   spirit of Panigrahi & Gioux, *J. Biomed. Opt.* 24(7), 071606 (2019).
6. **Synthetic burn phantom** (`src/burn_phantom.py`): generates a
   plausible (but synthetic) burn-severity map and corresponding optical
   property fields for the demo pipeline to run on.

## Repository layout

```
porcine-burn-sfdi/
├── src/
│   ├── sfdi_demodulation.py     # three-phase AC/DC demodulation + calibration
│   ├── diffusion_model.py       # forward model + LUT inversion
│   ├── monte_carlo.py           # vectorized MC validator
│   ├── ml_inversion.py          # MLP-based inversion
│   └── burn_phantom.py          # synthetic severity/optical-property maps
├── matlab/
│   ├── sfdiDemodulate.m
│   ├── diffuseReflectance.m
│   ├── invertOpticalProperties.m
│   └── run_demo.m
├── scripts/run_demo.py          # end-to-end Python demo -> figures/demo_summary.png
├── tests/test_sfdi.py           # pytest unit tests
├── figures/                     # demo output (git-ignored except .gitkeep)
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py
pytest tests/
```

`run_demo.py` generates a synthetic burn phantom, forward-simulates
noisy multi-frequency SFDI data, recovers optical properties via both
the lookup table and the ML regressor, cross-checks the diffusion model
against the Monte Carlo simulator at a few sample points, and saves a
summary figure to `figures/demo_summary.png`.

For the MATLAB version, add `matlab/` to your path and run
`matlab/run_demo.m`.

## Citation

If you build on this code, please cite the associated publication above.
This repository itself can be cited as:

```
Ahmmed, R. (2026). SFDI Burn-Severity Optical Property Mapping [Software].
https://github.com/raselece25/porcine-burn-sfdi
```

## License

MIT — see [LICENSE](LICENSE).
