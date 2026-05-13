import time
import numpy as np
from scipy.optimize import linprog
from scipy import sparse


def build_transport_lp(C, a, b):
    """
    Convert the transport problem into standard LP form.

    min c^T x
    s.t. A_eq x = b_eq
         x >= 0

    where x = vec(P).
    """
    n, m = C.shape

    if len(a) != n:
        raise ValueError("Length of a must match number of rows of C.")
    if len(b) != m:
        raise ValueError("Length of b must match number of columns of C.")

    c = C.reshape(-1)

    # Row constraints: sum_j P[i,j] = a[i]
    row_blocks = []
    for i in range(n):
        row = sparse.lil_matrix((1, n * m))
        for j in range(m):
            row[0, i * m + j] = 1.0
        row_blocks.append(row)

    A_rows = sparse.vstack(row_blocks)

    # Column constraints: sum_i P[i,j] = b[j]
    col_blocks = []
    for j in range(m):
        row = sparse.lil_matrix((1, n * m))
        for i in range(n):
            row[0, i * m + j] = 1.0
        col_blocks.append(row)

    A_cols = sparse.vstack(col_blocks)

    A_eq = sparse.vstack([A_rows, A_cols]).tocsr()
    b_eq = np.concatenate([a, b])

    bounds = [(0, None)] * (n * m)

    return c, A_eq, b_eq, bounds


def solve_lp_scipy(C, a, b, method):
    """
    Solve OT LP using scipy.optimize.linprog.

    method options:
    - "highs-ds"  : dual simplex
    - "highs-ipm" : interior point method
    """
    n, m = C.shape

    c, A_eq, b_eq, bounds = build_transport_lp(C, a, b)

    start = time.perf_counter()

    result = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
    )

    elapsed = time.perf_counter() - start

    if not result.success:
        raise RuntimeError(f"LP solve failed with method={method}: {result.message}")

    P = result.x.reshape(n, m)

    return {
        "P": P,
        "objective": float(result.fun),
        "time": elapsed,
        "status": result.message,
        "method": method,
    }


def solve_lp_simplex(C, a, b):
    """
    Solve using HiGHS dual simplex.
    """
    return solve_lp_scipy(C, a, b, method="highs-ds")


def solve_lp_ipm(C, a, b):
    """
    Solve using HiGHS interior point method.
    """
    return solve_lp_scipy(C, a, b, method="highs-ipm")

def solve_lp_pdhg(C, a, b, max_iter=20000, tau=0.1, sigma=0.1, tol=1e-6):
    """
    Simple NumPy implementation of PDHG for the OT LP.

    This follows the update rule from the assignment.

    P update:
        P^{k+1} = max(0, P^k - tau * (C - f 1^T - 1 g^T))

    Extrapolation:
        P_bar = 2 P^{k+1} - P^k

    Dual update:
        f^{k+1} = f^k + sigma * (P_bar 1 - a)
        g^{k+1} = g^k + sigma * (P_bar^T 1 - b)
    """
    start = time.perf_counter()

    n, m = C.shape

    P = np.ones((n, m)) / (n * m)
    f = np.zeros(n)
    g = np.zeros(m)

    history = []

    for k in range(max_iter):
        P_old = P.copy()

        gradient = C - f[:, None] - g[None, :]
        P = np.maximum(0.0, P - tau * gradient)

        P_bar = 2.0 * P - P_old

        f = f + sigma * (a - P_bar.sum(axis=1))
        g = g + sigma * (b - P_bar.sum(axis=0))

        if k % 100 == 0 or k == max_iter - 1:
            violation = (
                np.linalg.norm(P.sum(axis=1) - a, ord=1)
                + np.linalg.norm(P.sum(axis=0) - b, ord=1)
            )
            history.append((k, violation))

            if violation < tol:
                break

    elapsed = time.perf_counter() - start

    objective = float(np.sum(C * P))

    return {
        "P": P,
        "objective": objective,
        "time": elapsed,
        "status": f"PDHG finished after {k + 1} iterations",
        "method": "pdhg",
        "iterations": k + 1,
        "history": history,
    }