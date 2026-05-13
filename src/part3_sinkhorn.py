import numpy as np
import pandas as pd

from src.config import (
    MU_SOURCE,
    COV_SOURCE,
    MU_TARGET,
    COV_TARGET,
    RANDOM_SEED,
    TABLE_DIR,
    FIGURE_DIR,
)
from src.data import make_gaussian_clouds, make_uniform_weights
from src.costs import squared_euclidean_cost
from src.lp_solvers import solve_lp_ipm
from src.sinkhorn_solvers import (
    solve_sinkhorn_pot,
    solve_sinkhorn_manual,
    solve_sinkhorn_log_pot,
)
from src.metrics import transport_cost, marginal_violation
from src.plotting import plot_regularization_path, plot_sinkhorn_convergence


def run_sinkhorn_comparison():
    """
    Compare POT Sinkhorn and manual Sinkhorn for n = 200.
    """
    n = 200
    lam = 1.0

    print("=" * 70)
    print(f"Sinkhorn comparison: n={n}, lambda={lam}")

    X, Y = make_gaussian_clouds(
        n=n,
        mu_source=MU_SOURCE,
        cov_source=COV_SOURCE,
        mu_target=MU_TARGET,
        cov_target=COV_TARGET,
        seed=RANDOM_SEED,
    )

    a, b = make_uniform_weights(n)
    C = squared_euclidean_cost(X, Y)

    rows = []

    solvers = [
        ("pot_sinkhorn", solve_sinkhorn_pot),
        ("manual_sinkhorn", solve_sinkhorn_manual),
    ]

    for method_name, solver in solvers:
        print(f"Solving with {method_name}...")

        result = solver(C, a, b, lam)
        P = result["P"]

        rows.append({
            "n": n,
            "lambda": lam,
            "method": method_name,
            "time_seconds": result["time"],
            "regularized_objective": result["objective"],
            "transport_cost": transport_cost(C, P),
            "marginal_violation": marginal_violation(P, a, b),
            "status": result["status"],
        })

        print(f"  time: {result['time']:.4f}")
        print(f"  regularized objective: {result['objective']:.8f}")
        print(f"  transport cost: {transport_cost(C, P):.8f}")
        print(f"  violation: {marginal_violation(P, a, b):.3e}")

    df = pd.DataFrame(rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLE_DIR / "part3_sinkhorn_comparison.csv"
    df.to_csv(output_path, index=False)

    print("=" * 70)
    print(df)
    print(f"Saved table to: {output_path}")

    return df


def run_sinkhorn_regularization_path():
    """
    Solve entropic OT for lambda values from 10 to 0.01.
    """
    n = 200
    lambda_values = np.logspace(1, -2, 15)

    print("=" * 70)
    print(f"Sinkhorn regularization path: n={n}")

    X, Y = make_gaussian_clouds(
        n=n,
        mu_source=MU_SOURCE,
        cov_source=COV_SOURCE,
        mu_target=MU_TARGET,
        cov_target=COV_TARGET,
        seed=RANDOM_SEED,
    )

    a, b = make_uniform_weights(n)
    C = squared_euclidean_cost(X, Y)

    lp_result = solve_lp_ipm(C, a, b)
    w_lp = transport_cost(C, lp_result["P"])

    print(f"LP ground truth cost: {w_lp:.8f}")

    rows = []

    for lam in lambda_values:
        print(f"Solving lambda={lam:.4e} with manual Sinkhorn...")

        result = solve_sinkhorn_log_pot(C, a, b, lam, num_iter=100000, stop_thresh=1e-9)
        P = result["P"]

        cost = transport_cost(C, P)
        violation = marginal_violation(P, a, b)

        rows.append({
            "n": n,
            "lambda": lam,
            "method": "pot_sinkhorn_log",
            "regularized_objective": result["objective"],
            "transport_cost": cost,
            "lp_cost": w_lp,
            "cost_minus_lp": cost - w_lp,
            "marginal_violation": violation,
            "time_seconds": result["time"],
            "status": result["status"],
        })

        print(f"  transport cost: {cost:.8f}")
        print(f"  cost - LP: {cost - w_lp:.3e}")
        print(f"  violation: {violation:.3e}")

    df = pd.DataFrame(rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLE_DIR / "part3_sinkhorn_regularization_path.csv"
    df.to_csv(output_path, index=False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / "part3_sinkhorn_regularization_path.png"
    plot_regularization_path(
        df=df,
        lambda_col="lambda",
        cost_col="transport_cost",
        reference_cost=w_lp,
        output_path=figure_path,
        title="Entropic regularization path",
    )

    print("=" * 70)
    print(df)
    print(f"Saved table to: {output_path}")
    print(f"Saved figure to: {figure_path}")

    return df

def run_sinkhorn_convergence_plot():
    """
    Plot marginal violation vs iteration for lambda = 1, 0.1, 0.01.
    Uses manual Sinkhorn to expose iteration history.
    """
    n = 200
    lambda_values = [1.0, 0.1, 0.01]

    print("=" * 70)
    print(f"Sinkhorn convergence plot: n={n}")

    X, Y = make_gaussian_clouds(
        n=n,
        mu_source=MU_SOURCE,
        cov_source=COV_SOURCE,
        mu_target=MU_TARGET,
        cov_target=COV_TARGET,
        seed=RANDOM_SEED,
    )

    a, b = make_uniform_weights(n)
    C = squared_euclidean_cost(X, Y)

    histories = {}
    rows = []

    for lam in lambda_values:
        print(f"Running manual Sinkhorn for lambda={lam}...")

        result = solve_sinkhorn_manual(
            C,
            a,
            b,
            lam,
            num_iter=10000,
            stop_thresh=1e-9,
        )

        P = result["P"]
        violation = marginal_violation(P, a, b)

        label = f"lambda={lam}"
        histories[label] = result["history"]

        rows.append({
            "n": n,
            "lambda": lam,
            "method": "manual_sinkhorn",
            "iterations": result["iterations"],
            "final_marginal_violation": violation,
            "time_seconds": result["time"],
            "status": result["status"],
        })

        print(f"  iterations: {result['iterations']}")
        print(f"  final violation: {violation:.3e}")

    df = pd.DataFrame(rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLE_DIR / "part3_sinkhorn_convergence.csv"
    df.to_csv(output_path, index=False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / "part3_sinkhorn_convergence.png"
    plot_sinkhorn_convergence(histories, figure_path)

    print("=" * 70)
    print(df)
    print(f"Saved table to: {output_path}")
    print(f"Saved figure to: {figure_path}")

    return df

def main():
    run_sinkhorn_comparison()
    run_sinkhorn_regularization_path()
    run_sinkhorn_convergence_plot()


if __name__ == "__main__":
    main()