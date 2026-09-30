import numpy as np


class RBF:
    """Radial basis function kernel with ARD lengthscales.

    k(x, x') = sigma_f^2 * exp(-0.5 * sum_j ((x_j - x'_j)^2 / l_j^2))
    """

    def __init__(self, lengthscale=1.0, sigma_f=1.0):
        self.lengthscale = lengthscale
        self.sigma_f = sigma_f

    def __call__(self, X1, X2):
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)
        if X1.ndim == 1:
            X1 = X1[None, :]
        if X2.ndim == 1:
            X2 = X2[None, :]

        l = np.asarray(self.lengthscale, dtype=np.float64)
        if l.ndim == 0:
            l = np.full(X1.shape[1], float(l))

        X1_scaled = X1 / l
        X2_scaled = X2 / l

        sq1 = np.sum(X1_scaled ** 2, axis=1, keepdims=True)
        sq2 = np.sum(X2_scaled ** 2, axis=1, keepdims=True)
        cross = X1_scaled @ X2_scaled.T
        sq_dist = sq1 + sq2.T - 2 * cross
        sq_dist = np.maximum(sq_dist, 0.0)

        return (self.sigma_f ** 2) * np.exp(-0.5 * sq_dist)

    def diag(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        return np.full(X.shape[0], self.sigma_f ** 2)