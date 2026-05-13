import pandas as pd

from src.config import (
    MU_SOURCE,
    COV_SOURCE,
    MU_TARGET,
    COV_TARGET,
    RANDOM_SEED,
    N_VALUES_DEBUG,
    N_VALUES_MAIN,
    TABLE_DIR,
    FIGURE_DIR,
)
from src.data import make_gaussian_clouds, make_uniform_weights
from src.costs import squared_euclidean_cost
from src.lp_solvers import solve_lp_simplex, solve_lp_ipm, solve_lp_pdhg
from src.metrics import transport_cost, marginal_violation, check_transport_plan
from src.plotting import plot_timing_loglog


def run_small_correctness_checks():
    """
    Verify correctness for n = 3 and n = 4.
    """
    for n in N_VALUES_DEBUG:
        print("=" * 70)
        print(f"Small correctness check: n = {n}")

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

        simplex = solve_lp_simplex(C, a, b)
        ipm = solve_lp_ipm(C, a, b)

        print("simplex objective:", simplex["objective"])
        print("ipm objective:", ipm["objective"])
        print("difference:", abs(simplex["objective"] - ipm["objective"]))

        print("\nSimplex feasibility:")
        check_transport_plan(simplex["P"], a, b)

        print("\nIPM feasibility:")
        check_transport_plan(ipm["P"], a, b)

        print("\nmanual cost simplex:", transport_cost(C, simplex["P"]))
        print("manual cost ipm:", transport_cost(C, ipm["P"]))

        print()


def run_main_lp_experiment():
    """
    Run LP timing and accuracy experiments for required n values.
    """
    rows = []

    solvers = [
    ("simplex", solve_lp_simplex),
    ("ipm", solve_lp_ipm),
    ("pdhg", lambda C, a, b: solve_lp_pdhg(
        C,
        a,
        b,
        max_iter=20000,
        tau=0.01,
        sigma=5.0,
        tol=1e-6,
    )),
    ]

    for n in N_VALUES_MAIN:
        print("=" * 70)
        print(f"Main LP experiment: n = {n}")

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

        for method_name, solver in solvers:
            print(f"Solving with {method_name}...")

            result = solver(C, a, b)
            P = result["P"]

            cost = transport_cost(C, P)
            violation = marginal_violation(P, a, b)

            rows.append({
                "n": n,
                "method": method_name,
                "time_seconds": result["time"],
                "transport_cost": cost,
                "marginal_violation": violation,
                "solver_status": result["status"],
            })

            print(f"  time: {result['time']:.4f} sec")
            print(f"  cost: {cost:.8f}")
            print(f"  violation: {violation:.3e}")

    df = pd.DataFrame(rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLE_DIR / "part1_lp_results.csv"
    df.to_csv(output_path, index=False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / "part1_lp_timing_loglog.png"
    plot_timing_loglog(df, figure_path)

    print("=" * 70)
    print("Part 1 LP results:")
    print(df)
    print(f"\nSaved table to: {output_path}")
    print(f"Saved timing plot to: {figure_path}")

    return df


def main():
    run_small_correctness_checks()
    run_main_lp_experiment()


if __name__ == "__main__":
    main()