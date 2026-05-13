import numpy as np
import pandas as pd

from src.config import (
    MU_SOURCE,
    COV_SOURCE,
    MU_TARGET,
    COV_TARGET,
    TABLE_DIR,
    FIGURE_DIR,
)
from src.data import make_gaussian_clouds, make_uniform_weights
from src.costs import squared_euclidean_cost
from src.lp_solvers import solve_lp_ipm
from src.gaussian_ot import gaussian_w2_squared
from src.metrics import transport_cost
from src.plotting import (
    plot_gaussian_convergence,
    plot_gaussian_transport_arrows,
)


def run_gaussian_convergence_experiment():
    """
    Compare discrete OT costs to closed-form Gaussian W2^2.
    """
    n_values = [50, 100, 200, 500]
    seeds = list(range(10))

    w2_closed_form = gaussian_w2_squared(
        MU_SOURCE,
        COV_SOURCE,
        MU_TARGET,
        COV_TARGET,
    )

    print("=" * 70)
    print(f"Closed-form Gaussian W2^2: {w2_closed_form:.8f}")

    rows = []

    for n in n_values:
        for seed in seeds:
            print(f"Solving discrete OT: n={n}, seed={seed}")

            X, Y = make_gaussian_clouds(
                n=n,
                mu_source=MU_SOURCE,
                cov_source=COV_SOURCE,
                mu_target=MU_TARGET,
                cov_target=COV_TARGET,
                seed=seed,
            )

            a, b = make_uniform_weights(n)
            C = squared_euclidean_cost(X, Y)

            result = solve_lp_ipm(C, a, b)
            cost = transport_cost(C, result["P"])

            rows.append({
                "n": n,
                "seed": seed,
                "method": "lp_ipm",
                "discrete_ot_cost": cost,
                "gaussian_w2_squared": w2_closed_form,
                "error": cost - w2_closed_form,
                "time_seconds": result["time"],
            })

            print(f"  cost: {cost:.8f}")
            print(f"  error: {cost - w2_closed_form:.3e}")

    raw_df = pd.DataFrame(rows)

    summary_df = (
        raw_df
        .groupby("n")
        .agg(
            mean_cost=("discrete_ot_cost", "mean"),
            std_cost=("discrete_ot_cost", "std"),
            mean_error=("error", "mean"),
            std_error=("error", "std"),
            mean_time=("time_seconds", "mean"),
        )
        .reset_index()
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = TABLE_DIR / "part4_gaussian_raw.csv"
    summary_path = TABLE_DIR / "part4_gaussian_summary.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / "part4_gaussian_convergence.png"

    plot_gaussian_convergence(
        summary_df=summary_df,
        reference_cost=w2_closed_form,
        output_path=figure_path,
    )

    print("=" * 70)
    print(summary_df)
    print(f"Saved raw table to: {raw_path}")
    print(f"Saved summary table to: {summary_path}")
    print(f"Saved figure to: {figure_path}")

    return raw_df, summary_df

def run_gaussian_arrow_visualization():
    """
    For n = 200, plot OT arrows over Gaussian contour ellipses.
    """
    n = 200
    seed = 0

    print("=" * 70)
    print(f"Gaussian arrow visualization: n={n}, seed={seed}")

    X, Y = make_gaussian_clouds(
        n=n,
        mu_source=MU_SOURCE,
        cov_source=COV_SOURCE,
        mu_target=MU_TARGET,
        cov_target=COV_TARGET,
        seed=seed,
    )

    a, b = make_uniform_weights(n)
    C = squared_euclidean_cost(X, Y)

    result = solve_lp_ipm(C, a, b)
    P = result["P"]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "part4_gaussian_arrows.png"

    plot_gaussian_transport_arrows(
        X=X,
        Y=Y,
        P=P,
        mu_source=MU_SOURCE,
        cov_source=COV_SOURCE,
        mu_target=MU_TARGET,
        cov_target=COV_TARGET,
        output_path=output_path,
        max_edges=120,
    )

    print(f"Saved figure to: {output_path}")

def main():
    run_gaussian_convergence_experiment()
    run_gaussian_arrow_visualization()


if __name__ == "__main__":
    main()