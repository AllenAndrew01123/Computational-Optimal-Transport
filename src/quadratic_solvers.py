import time
import numpy as np
import cvxpy as cp


def solve_quadratic_cvxpy(C, a, b, lam, solver, solver_options=None):
    """
    Solve quadratically regularized OT using CVXPY.

    Problem:
        min <C, P> + (lam / 2) ||P||_F^2
        s.t. P 1 = a
             P^T 1 = b
             P >= 0

    Parameters
    ----------
    C : np.ndarray
        Cost matrix, shape (n, n).
    a : np.ndarray
        Source weights, shape (n,).
    b : np.ndarray
        Target weights, shape (n,).
    lam : float
        Quadratic regularization parameter.
    solver : str
        CVXPY solver name.
    solver_options : dict
        Extra options passed to CVXPY solve.

    Returns
    -------
    dict
        Contains P, objective, time, status, solver.
    """
    if solver_options is None:
        solver_options = {}

    n, m = C.shape

    P = cp.Variable((n, m), nonneg=True)

    objective = cp.sum(cp.multiply(C, P)) + (lam / 2.0) * cp.sum_squares(P)

    constraints = [
        cp.sum(P, axis=1) == a,
        cp.sum(P, axis=0) == b,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    start = time.perf_counter()
    problem.solve(solver=solver, **solver_options)
    elapsed = time.perf_counter() - start

    if P.value is None:
        raise RuntimeError(f"Quadratic solve failed with solver={solver}, status={problem.status}")

    return {
        "P": np.asarray(P.value),
        "objective": float(problem.value),
        "time": elapsed,
        "status": problem.status,
        "solver": solver,
        "lambda": lam,
    }


def solve_quadratic_osqp(C, a, b, lam, rho=None):
    """
    Solve quadratic OT using OSQP.

    OSQP is ADMM-based.
    """
    options = {
    "verbose": False,
    "eps_abs": 1e-8,
    "eps_rel": 1e-8,
    "max_iter": 100000,
    }

    if rho is not None:
        options["rho"] = rho

    return solve_quadratic_cvxpy(
        C=C,
        a=a,
        b=b,
        lam=lam,
        solver=cp.OSQP,
        solver_options=options,
    )


def solve_quadratic_ipm(C, a, b, lam):
    """
    Solve quadratic OT using CLARABEL through CVXPY.

    CLARABEL is an interior-point conic solver available with modern CVXPY.
    """
    return solve_quadratic_cvxpy(
        C=C,
        a=a,
        b=b,
        lam=lam,
        solver=cp.CLARABEL,
        solver_options={"verbose": False},
    )