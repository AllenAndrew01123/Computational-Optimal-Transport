import numpy as np
from scipy.linalg import sqrtm


def gaussian_w2_squared(mu1, cov1, mu2, cov2):
    """
    Closed-form squared W2 distance between two Gaussians.

    W2^2 = ||mu1 - mu2||^2
           + tr(cov1 + cov2 - 2 * (cov1^{1/2} cov2 cov1^{1/2})^{1/2})
    """
    mean_term = np.sum((mu1 - mu2) ** 2)

    cov1_sqrt = sqrtm(cov1)
    middle = cov1_sqrt @ cov2 @ cov1_sqrt
    middle_sqrt = sqrtm(middle)

    covariance_term = np.trace(cov1 + cov2 - 2.0 * middle_sqrt)

    value = mean_term + covariance_term

    return float(np.real(value))