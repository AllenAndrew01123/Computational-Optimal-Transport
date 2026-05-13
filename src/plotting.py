import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_timing_loglog(df, output_path):
    """
    Plot solve time vs n on a log-log scale.
    """
    plt.figure()

    for method in df["method"].unique():
        sub = df[df["method"] == method]
        plt.loglog(
            sub["n"],
            sub["time_seconds"],
            marker="o",
            label=method,
        )

    plt.xlabel("n")
    plt.ylabel("time (seconds)")
    plt.title("LP OT solve time vs problem size")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_regularization_path(df, lambda_col, cost_col, reference_cost, output_path, title):
    """
    Plot transport cost versus lambda on a log-scaled x-axis.
    """
    plt.figure()

    sorted_df = df.sort_values(lambda_col)

    plt.semilogx(
        sorted_df[lambda_col],
        sorted_df[cost_col],
        marker="o",
        label="regularized transport cost",
    )

    plt.axhline(
        reference_cost,
        linestyle="--",
        label="LP cost",
    )

    plt.xlabel("lambda")
    plt.ylabel("transport cost <C, P>")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
def plot_sinkhorn_convergence(histories, output_path):
    """
    Plot marginal violation vs Sinkhorn iteration.
    histories is a dict: label -> list of (iteration, violation).
    """
    plt.figure()

    for label, history in histories.items():
        iterations = [item[0] for item in history]
        violations = [item[1] for item in history]

        plt.semilogy(iterations, violations, marker="o", label=label)

    plt.xlabel("iteration")
    plt.ylabel("marginal violation")
    plt.title("Sinkhorn marginal violation vs iteration")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_coupling_arrows(X, Y, plans, output_path, max_edges=150):
    """
    Plot source/target point clouds and strongest transport edges.

    Parameters
    ----------
    X : array, shape (n, 2)
        Source points.
    Y : array, shape (n, 2)
        Target points.
    plans : dict
        Mapping title -> transport plan P.
    output_path : path
        Where to save figure.
    max_edges : int
        Number of largest P_ij entries to draw.
    """
    num_plots = len(plans)

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    for ax, (title, P) in zip(axes, plans.items()):
        ax.scatter(X[:, 0], X[:, 1], s=20, label="source")
        ax.scatter(Y[:, 0], Y[:, 1], s=20, marker="x", label="target")

        flat_indices = P.ravel().argsort()[-max_edges:]

        n, m = P.shape

        for idx in flat_indices:
            i = idx // m
            j = idx % m

            weight = P[i, j]

            if weight <= 0:
                continue

            ax.plot(
                [X[i, 0], Y[j, 0]],
                [X[i, 1], Y[j, 1]],
                linewidth=0.5,
                alpha=0.35,
            )

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_gaussian_convergence(summary_df, reference_cost, output_path):
    """
    Plot mean discrete OT cost ± one standard deviation vs n.
    """
    plt.figure()

    plt.errorbar(
        summary_df["n"],
        summary_df["mean_cost"],
        yerr=summary_df["std_cost"],
        marker="o",
        capsize=4,
        label="discrete OT mean ± std",
    )

    plt.axhline(
        reference_cost,
        linestyle="--",
        label="closed-form Gaussian W2²",
    )

    plt.xlabel("n")
    plt.ylabel("transport cost")
    plt.title("Discrete OT convergence to Gaussian W2²")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_gaussian_ellipse(ax, mean, cov, n_std=2.0):
    """
    Plot covariance ellipse for a 2D Gaussian.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    width, height = 2 * n_std * np.sqrt(eigvals)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=2,
    )

    ax.add_patch(ellipse)

def plot_gaussian_transport_arrows(
    X,
    Y,
    P,
    mu_source,
    cov_source,
    mu_target,
    cov_target,
    output_path,
    max_edges=120,
):
    """
    Plot strongest OT arrows over Gaussian covariance ellipses.
    """
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    ax.scatter(X[:, 0], X[:, 1], s=20, label="source samples")
    ax.scatter(Y[:, 0], Y[:, 1], s=20, marker="x", label="target samples")

    plot_gaussian_ellipse(ax, mu_source, cov_source, n_std=2.0)
    plot_gaussian_ellipse(ax, mu_target, cov_target, n_std=2.0)

    flat_indices = P.ravel().argsort()[-max_edges:]
    n, m = P.shape

    for idx in flat_indices:
        i = idx // m
        j = idx % m

        if P[i, j] <= 0:
            continue

        ax.arrow(
            X[i, 0],
            X[i, 1],
            Y[j, 0] - X[i, 0],
            Y[j, 1] - X[i, 1],
            length_includes_head=True,
            head_width=0.04,
            linewidth=0.5,
            alpha=0.35,
        )

    ax.set_title("Discrete OT arrows over Gaussian contours")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()