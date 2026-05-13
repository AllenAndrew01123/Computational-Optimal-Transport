import numpy as np


def make_gaussian_clouds(n, mu_source, cov_source, mu_target, cov_target, seed):
    """
    Generate source and target point clouds in R^2.

    Parameters
    ----------
    n : int
        Number of source points and target points.
    mu_source : np.ndarray
        Mean of source Gaussian, shape (2,).
    cov_source : np.ndarray
        Covariance of source Gaussian, shape (2, 2).
    mu_target : np.ndarray
        Mean of target Gaussian, shape (2,).
    cov_target : np.ndarray
        Covariance of target Gaussian, shape (2, 2).
    seed : int
        Random seed.

    Returns
    -------
    X : np.ndarray
        Source points, shape (n, 2).
    Y : np.ndarray
        Target points, shape (n, 2).
    """
    rng = np.random.default_rng(seed)

    X = rng.multivariate_normal(mean=mu_source, cov=cov_source, size=n)
    Y = rng.multivariate_normal(mean=mu_target, cov=cov_target, size=n)

    return X, Y


def make_uniform_weights(n):
    """
    Create uniform probability weights.

    Returns
    -------
    a : np.ndarray
        Source weights, shape (n,).
    b : np.ndarray
        Target weights, shape (n,).
    """
    a = np.ones(n) / n
    b = np.ones(n) / n

    return a, b