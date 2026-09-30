import numpy as np
from sklearn.model_selection import train_test_split
from smallgp import SmallGPRegressor


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.normal(size=200)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=0
    )

    model = SmallGPRegressor()
    model.fit(X_tr, y_tr)
    model.fit_conformal(X_cal, y_cal, X_val, y_val, alpha=0.1, verbose=True)

    lo, hi = model.predict_interval_conformal(X_val, alpha=0.1)
    cov = float(np.mean((y_val >= lo) & (y_val <= hi)))
    width = float(np.mean(hi - lo))

    print(f"coverage: {cov:.3f}  width: {width:.3f}")
    assert cov >= 0.80, f"coverage {cov:.3f} < 0.80"


if __name__ == "__main__":
    test_conformal_coverage()
    print("all tests passed")