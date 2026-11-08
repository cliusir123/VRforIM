# Variance Reduction for Independent Metropolis

Code for the experiments of *Variance Reduction for Independent Metropolis*
(S. Liu, P. Dellaportas and M. K. Titsias) and its supplement.

## Requirements

Python 3.11 with

```
numpy scipy pandas matplotlib tqdm
jax==0.10.1 optax==0.2.8 dm-haiku==0.0.16 distrax==0.1.8
```

A GPU is optional and only used by the Gaussian process experiments. For it, also install
`jax-cuda12-plugin==0.10.1 jax-cuda12-pjrt==0.10.1` (with an older NVIDIA driver, add
`nvidia-cuda-nvcc-cu12` so that JAX compiles its kernels locally).

All commands below are run from the folder they refer to.

## Main paper

| Result | Folder | Command | Output |
|---|---|---|---|
| Figure 1 (a), (b): N(0, 1) target, N(0, σ²) proposal | `Simulation_exp/` | `python sim.py` | `vrf11.pdf`, `vrf12.pdf` |
| Figure 1 (c), (d): N(0, 1) target, t_ν proposal | `Simulation_exp/` | `python sim2.py` | `vrf21.pdf`, `vrf22.pdf` |
| Table 1: marginal likelihood under the mixture of g-priors | `gexp/` | `python run_gprior.py` | printed table |
| Tables 3, 4: logistic regression | `adaptive_experiments/` | `DATASETS="ripley pima heart german" ./run_all.sh` | `results/` |
| Table 5: Gaussian process hyperparameters | `adaptive_experiments/` | `DATASETS="boston pendulum" ./run_all.sh` | `results/` |

### Table 1 (`gexp/`)

- `data.csv` holds the synthetic data (N = 50, X_i ~ N(0, 1), Y ~ N(4X_3 + 4X_4, 2.5²));
  `python generate_data.py --seed 1 --out data.csv` draws a new data set.
- `proposals/x<model>.npy` are the fitted K = 4 mixture proposals (weights and g_k) used for the
  table. `python train_proposals.py` refits them (EM with several restarts, keeping the best
  fit; it overwrites `proposals/`).
- `python run_gprior.py` loads the proposals and prints, for each of the 15 models, the IMCV
  and crude Monte Carlo estimates, the numerical integration value and the VRF
  (100 replications of size 1000; about 3 minutes). `--out table.md` also writes the table,
  `--models 34 1234` runs a subset, `--seed` changes the random numbers.

For the models whose likelihood sits far in the tail of the prior (X34, X234, X1234), the
crude Monte Carlo estimator is very heavy-tailed: its variance over 100 replications depends
on whether a few rare prior draws are sampled, so the VRF of these models changes a lot with
the random seed while the IMCV estimate stays close to the numerical integration value.

### Tables 3, 4 and 5 (`adaptive_experiments/`)

- `run_adaptive.py` runs one experiment: burn-in adaptation of the Gaussian proposal by KL
  minimisation (DSVI gradients, Adam, cosine learning rate), then independent replications of
  - `--mode after`: IM with the adapted proposal held fixed (Table 3 and the IMCV(c), RB and
    CIM columns), or
  - `--mode during`: Algorithm 2, batch adaptive IM with `--batch-size` B and `--n-batches` ℓ
    (Tables 4 and 5),

  and prints the VRFs against the IM estimator. `python run_adaptive.py -h` lists all options.
- `checkpoints/<dataset>.npz` store the adapted proposals; `run_all.sh` reuses them, so a run
  skips the burn-in adaptation.
- `./run_all.sh` runs all the table settings in parallel (logistic regression on CPU, Gaussian
  processes on GPU); `DATASETS="pima heart" ./run_all.sh` restricts the data sets. Each run
  writes `results/runs/<table>_<dataset>_<setting>.log`, ending with its VRF table and the
  estimated coefficients, and the estimates in the matching `.npz`.
- The IM proposal covariance is the adapted one multiplied by 1.1 for Boston and
  Pendulum and left unchanged for the other data sets (`--cov-scale` overrides it).
- `results/tables.md` and `results/tables.tex` contain the tables as reported, and
  `results/hyperparams.json` the settings of the runs behind them.

Data sets are in `data/` (Ripley, Pima, Heart, German for logistic regression; Boston housing
and Pendulum for Gaussian process regression).

## Supplement

| Result | Folder | Command | Output |
|---|---|---|---|
| Section 3.1, Figure 1: IMCV against importance sampling with control variates | `compareIS/` | `python imvsis.py` | `mse_plot.pdf` |
| Section 3.2, Figure 2: the same comparison along an adaptation | `compareIS/` | `python adp.py` | `mse_vs_batch_size.pdf`, `mse_vs_num_batches.pdf` |
| Section 5, Table 1: Gaussian target of dimension d | `adaptive_experiments/` | `python run_adaptive.py --model gaussian --dim d --mode after` | printed table |
| Section 5, Figure 3: VRF against the dimension | `dimanalysis/` | `python test.py && python plot_vrf_boxplot.py` | `gaussianbox.pdf` |
