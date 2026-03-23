# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking project comparing Bayesian inference frameworks — Stan (CmdStanPy, PyStan), PyMC, and PyMC with JAX samplers (NumPyro, BlackJAX) — on a hierarchical Bradley-Terry tennis player skill model. Uses Jeff Sackmann's ATP match dataset. Accompanies a blog post at https://martiningram.github.io/mcmc-comparison/.

## Running Benchmarks

```bash
# Run all benchmarks across multiple dataset sizes (years: 1968–2020)
bash fit_all.sh <target_dir>

# Run individual fitting scripts (each takes a start year as argument)
python fit_pymc.py <year>
python fit_cmdstanpy.py <year>
python fit_pymc_numpyro.py <year>
python fit_pymc_blackjax.py <year>
```

Output is saved to organized directories: `{target_dir}/{framework}/` with runtime `.txt` files and ArviZ netCDF `.nc` sample files.

## Architecture

- **Data layer**: `sackmann.py` processes raw ATP CSV data; `fetch_data.py` provides `create_arrays()` for data prep and `get_pymc_model()` for model construction
- **Stan models**: `stan_model.stan` (vectorized) and `stan_model_optimised.stan` (explicit loop per Bob Carpenter's advice)
- **Fitting scripts**: Each `fit_*.py` script follows the same pattern — load data, create model, time sampling, save results
- **Orchestration**: `fit_all.sh` runs all frameworks across dataset sizes with various configurations (CPU/GPU, parallel/vectorized chains)
- **Analysis**: `Compare runtimes.ipynb` aggregates runtimes, computes ESS/second efficiency, and validates parameter consistency across methods

## Dependencies

Core packages in `requirements.txt`: pandas, numpy, scikit-learn, arviz, toolz. Framework-specific: pymc, cmdstanpy, blackjax, numpyro, jax. External data: clone `https://github.com/JeffSackmann/tennis_atp` separately.
