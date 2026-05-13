import numpy as np


def transport_cost(C, P):
    """
    Compute <C, P> = sum_ij C_ij P_ij.
    """
    return float(np.sum(C * P))


def marginal_violation(P, a, b):
    """
    Compute ||P1 - a||_1 + ||P^T1 - b||_1.
    """
    row_violation = np.linalg.norm(P.sum(axis=1) - a, ord=1)
    col_violation = np.linalg.norm(P.sum(axis=0) - b, ord=1)

    return float(row_violation + col_violation)


def check_transport_plan(P, a, b, tol=1e-7):
    """
    Check whether P is a valid coupling.

    Conditions:
    1. P_ij >= 0
    2. row sums equal a
    3. column sums equal b
    4. total mass equals 1
    """
    min_entry = P.min()
    total_mass = P.sum()
    row_violation = np.linalg.norm(P.sum(axis=1) - a, ord=1)
    col_violation = np.linalg.norm(P.sum(axis=0) - b, ord=1)
    total_violation = row_violation + col_violation

    print("min entry:", min_entry)
    print("total mass:", total_mass)
    print("row violation:", row_violation)
    print("column violation:", col_violation)
    print("total marginal violation:", total_violation)

    assert min_entry >= -tol
    assert abs(total_mass - 1.0) <= tol
    assert row_violation <= tol
    assert col_violation <= tol