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
from src.quadratic_solvers import solve_quadratic_osqp, solve_quadratic_ipm
from src.metrics import transport_cost, marginal_violation
from src.config import FIGURE_DIR
from src.plotting import plot_regularization_path


def run_quadratic_solver_comparison():
    """
    Compare ADMM/OSQP and IPM/CLARABEL for n = 200.
    """
    n = 200
    lam = 0.1

    print("=" * 70)
    print(f"Quadratic solver comparison: n={n}, lambda={lam}")

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
        ("osqp_admm", solve_quadratic_osqp),
        ("clarabel_ipm", solve_quadratic_ipm),
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
            "objective_regularized": result["objective"],
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
    output_path = TABLE_DIR / "part2_quadratic_solver_comparison.csv"
    df.to_csv(output_path, index=False)

    print("=" * 70)
    print(df)
    print(f"Saved table to: {output_path}")

    return df

def run_quadratic_regularization_path():
    """
    Solve quadratic OT for many lambda values and compare transport cost to LP optimum.
    """
    n = 200
    lambda_values = np.logspace(1, -3, 10)

    print("=" * 70)
    print(f"Quadratic regularization path: n={n}")

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
        print(f"Solving lambda={lam:.4e} with CLARABEL/IPM...")

        result = solve_quadratic_ipm(C, a, b, lam)
        P = result["P"]

        cost = transport_cost(C, P)
        violation = marginal_violation(P, a, b)

        rows.append({
            "n": n,
            "lambda": lam,
            "method": "clarabel_ipm",
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
    output_path = TABLE_DIR / "part2_quadratic_regularization_path.csv"
    df.to_csv(output_path, index=False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / "part2_quadratic_regularization_path.png"
    plot_regularization_path(
        df=df,
        lambda_col="lambda",
        cost_col="transport_cost",
        reference_cost=w_lp,
        output_path=figure_path,
        title="Quadratic regularization path",
    )

    print("=" * 70)
    print(df)
    print(f"Saved table to: {output_path}")
    print(f"Saved figure to: {figure_path}")

    return df

def run_quadratic_rho_sensitivity():
    """
    Test how OSQP's ADMM rho parameter affects solve time.
    """
    n = 200
    lam = 0.1
    rho_values = [0.01, 0.1, 1.0, 10.0]

    print("=" * 70)
    print(f"Quadratic rho sensitivity: n={n}, lambda={lam}")

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

    for rho in rho_values:
        print(f"Solving with OSQP rho={rho}...")

        result = solve_quadratic_osqp(C, a, b, lam, rho=rho)
        P = result["P"]

        rows.append({
            "n": n,
            "lambda": lam,
            "rho": rho,
            "method": "osqp_admm",
            "time_seconds": result["time"],
            "regularized_objective": result["objective"],
            "transport_cost": transport_cost(C, P),
            "marginal_violation": marginal_violation(P, a, b),
            "status": result["status"],
        })

        print(f"  time: {result['time']:.4f}")
        print(f"  transport cost: {transport_cost(C, P):.8f}")
        print(f"  violation: {marginal_violation(P, a, b):.3e}")

    df = pd.DataFrame(rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLE_DIR / "part2_quadratic_rho_sensitivity.csv"
    df.to_csv(output_path, index=False)

    print("=" * 70)
    print(df)
    print(f"Saved table to: {output_path}")

    return df

def main():
    run_quadratic_solver_comparison()
    run_quadratic_regularization_path()
    run_quadratic_rho_sensitivity()



if __name__ == "__main__":
    main()
