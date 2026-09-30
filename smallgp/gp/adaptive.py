import numpy as np


def adaptive_lengthscale(X, y=None, method="dim", min_ls=1e-3):
    """Initialize ARD lengthscales from data statistics.

    Methods:
      'dim'      — l_j = sqrt(d) for all j
      'half_dim' — l_j = d / 2 for all j
      'median'   — l_j = 1.4826 * median(|X[:, j] - median(X[:, j])|)
      'std'      — l_j = std(X[:, j])
      'range'    — l_j = (max - min) / 4
      'ones'     — l_j = 1
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    if method == "dim":
        l = np.full(d, np.sqrt(d))
    elif method == "half_dim":
        l = np.full(d, d / 2.0)
    elif method == "median":
        med = np.median(X, axis=0)
        l = 1.4826 * np.median(np.abs(X - med), axis=0)
    elif method == "std":
        l = np.std(X, axis=0)
    elif method == "range":
        l = (X.max(axis=0) - X.min(axis=0)) / 4.0
    elif method == "ones":
        l = np.ones(d)
    else:
        raise ValueError(f"Unknown method: {method}")

    l = np.where(np.isfinite(l) & (l > 0), l, 1.0)
    l = np.maximum(l, min_ls)
    return l