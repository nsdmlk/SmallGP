import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from .gp.kernels import RBF
from .gp.exact import ExactGP
from .gp.adaptive import adaptive_lengthscale
from .conformal.weighted import weighted_quantile, kernel_weights


class SmallGPRegressor(BaseEstimator, RegressorMixin):
    """Adaptive GP regressor for small nonlinear data.

    Point prediction: exact GP posterior mean.
    Prediction intervals: weighted conformal.
    """

    def __init__(self, lengthscale=None, sigma_f=1.0, noise=1e-2,
                 adaptive_ls=True, ls_method="dim", normalize=True,
                 optimize=True, n_restarts=2):
        self.lengthscale = lengthscale
        self.sigma_f = sigma_f
        self.noise = noise
        self.adaptive_ls = adaptive_ls
        self.ls_method = ls_method
        self.normalize = normalize
        self.optimize = optimize
        self.n_restarts = n_restarts

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        if self.normalize:
            self._x_scaler = StandardScaler().fit(X)
            self._y_mean = float(y.mean())
            self._y_std = float(y.std()) if y.std() > 0 else 1.0
            X = self._x_scaler.transform(X)
            y = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0

        if self.lengthscale is None and self.adaptive_ls:
            ls = adaptive_lengthscale(X, y, method=self.ls_method)
        elif self.lengthscale is None:
            ls = np.ones(X.shape[1])
        else:
            ls = np.asarray(self.lengthscale, dtype=np.float64)

        self._lengthscale_ = ls
        kernel = RBF(lengthscale=ls, sigma_f=self.sigma_f)
        self._gp = ExactGP(kernel, noise=self.noise)
        self._gp.fit(
            X, y,
            optimize=self.optimize,
            n_restarts=self.n_restarts,
        )
        self._lengthscale_ = self._gp.kernel.lengthscale
        return self

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=np.float64)
        if self.normalize:
            X = self._x_scaler.transform(X)

        if return_std:
            mu, std = self._gp.predict(X, return_std=True)
            mu = mu * self._y_std + self._y_mean
            std = std * self._y_std
            return mu, std
        else:
            mu = self._gp.predict(X)
            return mu * self._y_std + self._y_mean

    # ---------------- weighted conformal ----------------

    def fit_conformal(self, X_cal, y_cal, X_val, y_val,
                      alpha=0.1, h_grid=None, verbose=False):
        X_cal = np.asarray(X_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.float64)
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)

        if h_grid is None:
            h_grid = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])

        # calibration scores
        y_cal_pred = self.predict(X_cal)
        scores_cal = np.abs(y_cal - y_cal_pred)

        # embeddings
        if self.normalize:
            X_cal_emb = self._x_scaler.transform(X_cal)
            X_val_emb = self._x_scaler.transform(X_val)
        else:
            X_cal_emb = X_cal
            X_val_emb = X_val

        # tune h_cal on validation
        best_h = float(h_grid[0])
        best_score = np.inf
        target = 1 - alpha
        for h in h_grid:
            lo, hi = self._interval_from_scores(
                X_val, y_val, X_cal_emb, scores_cal, X_val_emb, alpha, float(h)
            )
            cov = np.mean((y_val >= lo) & (y_val <= hi))
            width = np.mean(hi - lo)
            penalty = 0.0
            if cov < target - 0.02:
                penalty = (target - cov) * 10.0
            score = width + penalty
            if score < best_score:
                best_score = score
                best_h = float(h)

        self._h_cal = best_h
        self._X_cal_emb = X_cal_emb
        self._scores_cal = scores_cal
        self._alpha_conformal = alpha

        if verbose:
            print(f"[conformal] h_cal={best_h:.4f}  score={best_score:.4f}")
        return self

    def _interval_from_scores(self, X, y, X_cal_emb, scores_cal,
                              X_emb, alpha, h):
        n_cal = len(scores_cal)
        q_level = min(1.0, (1 - alpha) * (1 + 1 / n_cal))
        y_hat = self.predict(X)

        lo = np.zeros(len(X))
        hi = np.zeros(len(X))
        for i in range(len(X)):
            w = kernel_weights(X_emb[i:i + 1], X_cal_emb, h)[0]
            q_hat = weighted_quantile(scores_cal, q_level, sample_weight=w)
            lo[i] = y_hat[i] - q_hat
            hi[i] = y_hat[i] + q_hat
        return lo, hi

    def predict_interval_conformal(self, X, alpha=None):
        if not hasattr(self, "_scores_cal"):
            raise RuntimeError("Call fit_conformal() first.")
        if alpha is None:
            alpha = self._alpha_conformal

        X = np.asarray(X, dtype=np.float64)
        if self.normalize:
            X_emb = self._x_scaler.transform(X)
        else:
            X_emb = X

        return self._interval_from_scores(
            X, None, self._X_cal_emb, self._scores_cal,
            X_emb, alpha, self._h_cal
        )