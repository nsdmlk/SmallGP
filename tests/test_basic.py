import numpy as np
from smallgp import SmallGPRegressor


def test_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=50)

    model = SmallGPRegressor()
    model.fit(X, y)
    y_hat = model.predict(X)

    assert y_hat.shape == (50,)
    mae = float(np.mean(np.abs(y - y_hat)))
    print(f"train MAE: {mae:.4f}")
    assert mae < 0.5


def test_predict_std():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = X[:, 0] ** 2 + 0.1 * rng.normal(size=30)

    model = SmallGPRegressor()
    model.fit(X, y)
    mu, std = model.predict(X, return_std=True)

    assert mu.shape == (30,)
    assert std.shape == (30,)
    assert np.all(std > 0)
    print(f"mean std: {std.mean():.4f}")


if __name__ == "__main__":
    test_fit_predict()
    test_predict_std()
    print("all tests passed")