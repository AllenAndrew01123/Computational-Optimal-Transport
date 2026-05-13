import numpy as np


def squared_euclidean_cost(X, Y):
    """
    Build squared Euclidean cost matrix.

    Parameters
    ----------
    X : np.ndarray
        Source points, shape (n, d).
    Y : np.ndarray
        Target points, shape (n, d).

    Returns
    -------
    C : np.ndarray
        Cost matrix, shape (n, n), where C[i, j] = ||X[i] - Y[j]||^2.
    """
    diff = X[:, None, :] - Y[None, :, :]
    C = np.sum(diff ** 2, axis=2)

    return C