import time
import numpy as np
import ot


def solve_sinkhorn_pot(C, a, b, lam, num_iter=10000, stop_thresh=1e-9):
    """
    Solve entropic OT using POT's Sinkhorn implementation.

    POT solves:
        min <C, P> + reg * entropy_term

    Here reg = lam.
    """
    start = time.perf_counter()

    P = ot.sinkhorn(
        a,
        b,
        C,
        reg=lam,
        numItermax=num_iter,
        stopThr=stop_thresh,
        verbose=False,
    )

    elapsed = time.perf_counter() - start

    objective = float(
        np.sum(C * P)
        + lam * np.sum(P * np.log(np.maximum(P, 1e-300)))
    )

    return {
        "P": P,
        "objective": objective,
        "time": elapsed,
        "status": "finished",
        "solver": "pot_sinkhorn",
        "lambda": lam,
    }


def solve_sinkhorn_manual(C, a, b, lam, num_iter=10000, stop_thresh=1e-9):
    """
    Manual Sinkhorn implementation.

    K_ij = exp(-C_ij / lam)
    P = diag(u) K diag(v)

    Updates:
        u <- a / (K v)
        v <- b / (K^T u)
    """
    start = time.perf_counter()

    K = np.exp(-C / lam)

    n, m = C.shape
    u = np.ones(n)
    v = np.ones(m)

    history = []

    for k in range(num_iter):
        u = a / np.maximum(K @ v, 1e-300)
        v = b / np.maximum(K.T @ u, 1e-300)

        if k % 10 == 0 or k == num_iter - 1:
            P = (u[:, None] * K) * v[None, :]
            violation = (
                np.linalg.norm(P.sum(axis=1) - a, ord=1)
                + np.linalg.norm(P.sum(axis=0) - b, ord=1)
            )
            history.append((k, violation))

            if violation < stop_thresh:
                break

    P = (u[:, None] * K) * v[None, :]

    elapsed = time.perf_counter() - start

    objective = float(
        np.sum(C * P)
        + lam * np.sum(P * np.log(np.maximum(P, 1e-300)))
    )

    return {
        "P": P,
        "objective": objective,
        "time": elapsed,
        "status": f"finished after {k + 1} iterations",
        "solver": "manual_sinkhorn",
        "lambda": lam,
        "iterations": k + 1,
        "history": history,
    }
def solve_sinkhorn_log_pot(C, a, b, lam, num_iter=100000, stop_thresh=1e-9):
    """
    Solve entropic OT using POT's log-domain stabilized Sinkhorn.

    This is more stable for small lambda.
    """
    start = time.perf_counter()

    P = ot.bregman.sinkhorn_log(
        a,
        b,
        C,
        reg=lam,
        numItermax=num_iter,
        stopThr=stop_thresh,
        verbose=False,
    )

    elapsed = time.perf_counter() - start

    objective = float(
        np.sum(C * P)
        + lam * np.sum(P * np.log(np.maximum(P, 1e-300)))
    )

    return {
        "P": P,
        "objective": objective,
        "time": elapsed,
        "status": "finished",
        "solver": "pot_sinkhorn_log",
        "lambda": lam,
    }