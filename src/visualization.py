from src.config import (
    MU_SOURCE,
    COV_SOURCE,
    MU_TARGET,
    COV_TARGET,
    RANDOM_SEED,
    FIGURE_DIR,
)
from src.data import make_gaussian_clouds, make_uniform_weights
from src.costs import squared_euclidean_cost
from src.lp_solvers import solve_lp_ipm
from src.quadratic_solvers import solve_quadratic_ipm
from src.sinkhorn_solvers import solve_sinkhorn_log_pot
from src.plotting import plot_coupling_arrows


def main():
    """
    Visualize four couplings on the same point cloud:
    LP, quadratic lambda=0.1, Sinkhorn lambda=1, Sinkhorn lambda=0.01.
    """
    n = 200

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

    print("Solving LP...")
    lp = solve_lp_ipm(C, a, b)

    print("Solving quadratic lambda=0.1...")
    quad = solve_quadratic_ipm(C, a, b, lam=0.1)

    print("Solving Sinkhorn lambda=1...")
    sinkhorn_1 = solve_sinkhorn_log_pot(C, a, b, lam=1.0)

    print("Solving Sinkhorn lambda=0.01...")
    sinkhorn_small = solve_sinkhorn_log_pot(C, a, b, lam=0.01)

    plans = {
        "LP": lp["P"],
        "Quadratic lambda=0.1": quad["P"],
        "Sinkhorn lambda=1": sinkhorn_1["P"],
        "Sinkhorn lambda=0.01": sinkhorn_small["P"],
    }

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "coupling_visualization.png"

    plot_coupling_arrows(
        X=X,
        Y=Y,
        plans=plans,
        output_path=output_path,
        max_edges=80,
    )

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()