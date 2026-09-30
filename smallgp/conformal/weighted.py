import numpy as np


def weighted_quantile(values, quantile, sample_weight=None):
    """Weighted quantile. Same convention as SmallMLP."""
    values = np.asarray(values, dtype=np.float64)
    if sample_weight is None:
        sample_weight = np.ones_like(values)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)

    if sample_weight.sum() <= 0:
        return float(np.quantile(values, quantile))

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return float(np.interp(quantile, weighted_quantiles, values))


def kernel_weights(X_query, X_cal, h):
    """Gaussian kernel weights for conformal.

    w_j(x*) = exp(-||x* - x_j||^2 / (2 h^2))
    Returns array of shape (n_query, n_cal).
    """
    X_query = np.asarray(X_query, dtype=np.float64)
    X_cal = np.asarray(X_cal, dtype=np.float64)
    if X_query.ndim == 1:
        X_query = X_query[None, :]
    if X_cal.ndim == 1:
        X_cal = X_cal[None, :]

    sq_q = np.sum(X_query ** 2, axis=1, keepdims=True)
    sq_c = np.sum(X_cal ** 2, axis=1, keepdims=True)
    cross = X_query @ X_cal.T
    sq_dist = sq_q + sq_c.T - 2 * cross
    sq_dist = np.maximum(sq_dist, 0.0)

    return np.exp(-sq_dist / (2 * max(h, 1e-9) ** 2))