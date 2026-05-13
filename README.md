# Computational Optimal Transport

IEOR 6616: Convex Optimization  
Programming Project: Computational Optimal Transport

## Overview

This project studies computational optimal transport using several solver paradigms and regularization methods.

The experiments use two random 2D Gaussian point clouds:

- Source samples: \(X_i \sim \mathcal{N}(\mu_1, \Sigma_1)\)
- Target samples: \(Y_j \sim \mathcal{N}(\mu_2, \Sigma_2)\)
- Uniform weights: \(a = b = \frac{1}{n}\mathbf{1}\)
- Cost matrix: \(C_{ij} = \|X_i - Y_j\|_2^2\)

The project compares:

- Linear programming OT
- Quadratic regularized OT
- Entropic regularized OT / Sinkhorn
- Discrete OT convergence to closed-form Gaussian \(W_2^2\)

## Team

Name: Allen Andrew  
UNI: TODO

Add teammate name and UNI here if applicable.

## Project structure

```txt
src/
  __init__.py
  config.py
  data.py
  costs.py
  metrics.py
  plotting.py

  lp_solvers.py
  quadratic_solvers.py
  sinkhorn_solvers.py
  gaussian_ot.py

  part1_lp.py
  part2_quadratic.py
  part3_sinkhorn.py
  part4_gaussian.py
  visualization.py
  run_all.py

outputs/
  figures/
  tables/

README.md
requirements.txt
report_notes.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

The required packages are listed in `requirements.txt`.

## Reproducibility

All main experiment parameters are stored in:

```txt
src/config.py
```

This includes:

- random seed
- Gaussian means
- Gaussian covariance matrices
- problem sizes
- output paths

The selected Gaussian parameters satisfy the assignment requirements:

- points are in \(\mathbb{R}^2\)
- means are distinct
- covariance matrices are non-diagonal
- weights are uniform

## Run experiments

### Part 1: LP optimal transport

Runs:

- small correctness checks for \(n=3,4\)
- LP solve times for \(n = 50,100,200,500\)
- simplex, IPM, and manual PDHG comparison
- transport cost and marginal violation table
- log-log timing plot

```bash
python -m src.part1_lp
```

Outputs:

```txt
outputs/tables/part1_lp_results.csv
outputs/figures/part1_lp_timing_loglog.png
```

### Part 2: Quadratic regularization

Solves:

\[
\min_{P \in U(a,b)} \langle C,P\rangle + \frac{\lambda}{2}\|P\|_F^2
\]

Runs:

- OSQP / ADMM vs CLARABEL / IPM comparison for \(n=200\)
- quadratic regularization path
- ADMM rho sensitivity experiment

```bash
python -m src.part2_quadratic
```

Outputs:

```txt
outputs/tables/part2_quadratic_solver_comparison.csv
outputs/tables/part2_quadratic_regularization_path.csv
outputs/tables/part2_quadratic_rho_sensitivity.csv
outputs/figures/part2_quadratic_regularization_path.png
```

### Part 3: Entropic regularization / Sinkhorn

Solves:

\[
\min_{P \in U(a,b)} \langle C,P\rangle + \lambda \sum_{ij} P_{ij}\log P_{ij}
\]

Runs:

- POT Sinkhorn vs manual Sinkhorn comparison
- entropic regularization path for 15 values of \(\lambda\)
- marginal violation vs iteration for \(\lambda = 1, 0.1, 0.01\)

The regularization path uses POT's log-domain Sinkhorn for numerical stability at small \(\lambda\).

```bash
python -m src.part3_sinkhorn
```

Outputs:

```txt
outputs/tables/part3_sinkhorn_comparison.csv
outputs/tables/part3_sinkhorn_regularization_path.csv
outputs/tables/part3_sinkhorn_convergence.csv
outputs/figures/part3_sinkhorn_regularization_path.png
outputs/figures/part3_sinkhorn_convergence.png
```

### Coupling visualization

Creates side-by-side visualizations of:

- LP coupling
- quadratic coupling with \(\lambda = 0.1\)
- Sinkhorn coupling with \(\lambda = 1\)
- Sinkhorn coupling with \(\lambda = 0.01\)

```bash
python -m src.visualization
```

Output:

```txt
outputs/figures/coupling_visualization.png
```

### Part 4: Gaussian optimal transport

Computes the closed-form squared Wasserstein-2 distance between two Gaussians:

\[
W_2^2(\mu_0,\nu_0)
=
\|\mu_1-\mu_2\|^2
+
\operatorname{tr}
\left(
\Sigma_1+\Sigma_2
-
2(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}
\right)
\]

Then compares it with discrete OT costs from sampled point clouds.

Runs:

- closed-form Gaussian \(W_2^2\)
- discrete OT for \(n = 50,100,200,500\)
- 10 random seeds per \(n\)
- mean ± one standard deviation plot
- Gaussian contour ellipse and OT arrow visualization for \(n=200\)

```bash
python -m src.part4_gaussian
```

Outputs:

```txt
outputs/tables/part4_gaussian_raw.csv
outputs/tables/part4_gaussian_summary.csv
outputs/figures/part4_gaussian_convergence.png
outputs/figures/part4_gaussian_arrows.png
```

## Solvers used

### LP optimal transport

- Simplex: `scipy.optimize.linprog(method="highs-ds")`
- Interior point method: `scipy.optimize.linprog(method="highs-ipm")`
- PDHG: manual NumPy implementation using the update equations from the assignment

### Quadratic regularized OT

- ADMM: CVXPY with OSQP
- Interior point / conic solver: CVXPY with CLARABEL

### Entropic regularized OT

- POT Sinkhorn
- manual NumPy Sinkhorn
- POT log-domain Sinkhorn for small \(\lambda\)

### Gaussian OT

- closed-form Gaussian \(W_2^2\)
- matrix square root computed with `scipy.linalg.sqrtm`

## Output files produced

Tables:

```txt
outputs/tables/part1_lp_results.csv
outputs/tables/part2_quadratic_solver_comparison.csv
outputs/tables/part2_quadratic_regularization_path.csv
outputs/tables/part2_quadratic_rho_sensitivity.csv
outputs/tables/part3_sinkhorn_comparison.csv
outputs/tables/part3_sinkhorn_regularization_path.csv
outputs/tables/part3_sinkhorn_convergence.csv
outputs/tables/part4_gaussian_raw.csv
outputs/tables/part4_gaussian_summary.csv
```

Figures:

```txt
outputs/figures/part1_lp_timing_loglog.png
outputs/figures/part2_quadratic_regularization_path.png
outputs/figures/part3_sinkhorn_regularization_path.png
outputs/figures/part3_sinkhorn_convergence.png
outputs/figures/coupling_visualization.png
outputs/figures/part4_gaussian_convergence.png
outputs/figures/part4_gaussian_arrows.png
```

## Notes on numerical behavior

The manual PDHG implementation is approximate. It reaches small marginal violation but does not solve the LP as exactly as simplex or IPM.

Naive Sinkhorn is extremely fast for moderate \(\lambda\), but it becomes unstable for small \(\lambda\) because the Gibbs kernel

\[
K_{ij} = \exp(-C_{ij}/\lambda)
\]

can underflow. For the small-\(\lambda\) regularization path, log-domain Sinkhorn is used.

The empirical Gaussian OT experiment has sampling variability because both source and target distributions are represented by finite random samples. The mean discrete cost is therefore not necessarily monotone in \(n\), but the results stay near the closed-form Gaussian \(W_2^2\).

## LLM usage

LLMs were used to help structure the code, debug implementation issues, and organize the report notes. The mathematical formulations, solver choices, experiments, and numerical outputs were checked against the assignment requirements.