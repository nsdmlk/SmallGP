import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class ExactGP:
    """Exact GP regression with Gaussian likelihood.

    Optimizes ARD lengthscales and noise via marginal likelihood
    maximization (L-BFGS-B).
    """

    def __init__(self, kernel, noise=1e-2):
        self.kernel = kernel
        self.noise = float(noise)
        self.X_train = None
        self.y_train = None
        self._cho = None
        self._alpha = None

    def fit(self, X, y, optimize=True, n_restarts=2):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.X_train = X
        self.y_train = y

        if optimize:
            self._optimize_hyperparams(X, y, n_restarts=n_restarts)

        self._recompute(X, y)
        return self

    def _recompute(self, X, y):
        n = X.shape[0]
        K = self.kernel(X, X) + (self.noise ** 2) * np.eye(n) + 1e-8 * np.eye(n)
        self._cho = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._cho, y)

    def _neg_log_mll(self, params, X, y):
        d = X.shape[1]
        ls = np.exp(params[:d])
        noise = np.exp(params[d])
        kernel = type(self.kernel)(
            lengthscale=ls, sigma_f=self.kernel.sigma_f
        )
        n = X.shape[0]
        K = kernel(X, X) + (noise ** 2) * np.eye(n) + 1e-8 * np.eye(n)
        try:
            cho = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = cho_solve(cho, y)
        L = cho[0]
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return 0.5 * (y @ alpha) + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)

    def _optimize_hyperparams(self, X, y, n_restarts=2):
        d = X.shape[1]
        ls0 = np.asarray(self.kernel.lengthscale, dtype=np.float64)
        if ls0.ndim == 0:
            ls0 = np.full(d, float(ls0))
        noise0 = self.noise

        def obj(params):
            return self._neg_log_mll(params, X, y)

        best_params = np.concatenate([np.log(ls0), [np.log(noise0)]])
        best_loss = obj(best_params)

        # first: optimize from init
        res = minimize(
            obj, best_params,
            method="L-BFGS-B",
            options={"maxiter": 100},
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

        # then: restarts with perturbations
        rng = np.random.default_rng(0)
        for _ in range(n_restarts):
            params0 = best_params + rng.normal(scale=0.5, size=best_params.shape)
            res = minimize(
                obj, params0,
                method="L-BFGS-B",
                options={"maxiter": 100},
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x

        ls_opt = np.exp(best_params[:d])
        noise_opt = float(np.exp(best_params[d]))
        self.kernel = type(self.kernel)(
            lengthscale=ls_opt, sigma_f=self.kernel.sigma_f
        )
        self.noise = noise_opt

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Ks = self.kernel(X, self.X_train)
        mu = Ks @ self._alpha
        if not return_std:
            return mu
        v = cho_solve(self._cho, Ks.T)
        Kss_diag = self.kernel.diag(X)
        var = np.maximum(Kss_diag - np.sum(Ks.T * v, axis=0), 1e-12)
        return mu, np.sqrt(var)

    def log_marginal_likelihood(self):
        if self._cho is None:
            raise RuntimeError("Call fit() first.")
        L = self._cho[0]
        n = len(self.y_train)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (self.y_train @ self._alpha) - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)