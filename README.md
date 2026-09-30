# SmallGP

> Gaussian Processes for small nonlinear data, with adaptive kernels and calibrated intervals.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-planning-orange.svg)]()

**SmallGP** is a library for **small nonlinear regression** ($n < 500$) that removes the two main obstacles of Gaussian Processes on small data: **manual kernel tuning** and **cubic scaling**. Kernel lengthscales are derived from dataset parameters ($n$, $d$), and a sparse approximation handles $n$ up to a few thousand without sacrificing calibration.

- **No kernel tuning.** Lengthscale initialized from data statistics.
- **Calibrated intervals.** Conformal wrapper on top of GP posterior.
- **Sparse variant** for $n > 500$ via inducing points.
- **Scikit-learn compatible API** (`fit` / `predict` / `predict_interval`).

---

## Why SmallGP?

Gaussian Processes are the natural choice for small nonlinear data — they provide calibrated uncertainty, work with few samples, and have strong theoretical foundations. But two things stop people from using them in practice:

| Problem                              | Standard GP | SmallGP |
| ------------------------------------ | ----------- | ------- |
| Kernel lengthscale must be tuned      | ✗           | ✓       |
| $O(n^3)$ scaling at fit time          | ✗           | ✓ (sparse) |
| Works with $n < 100$                  | ✓           | ✓       |
| Calibrated prediction intervals       | ✓           | ✓       |
| Scikit-learn API                      | ✗           | ✓       |

**Adaptive by construction.** Lengthscale, noise, and inducing point count are derived from dataset parameters — not searched.

---

## Status

**Planning stage.** This repository is a placeholder for future development.

Planned milestones:
- [ ] Adaptive lengthscale initialization from $n$, $d$, target scale
- [ ] Full GP baseline (exact inference)
- [ ] Sparse GP via inducing points ($m \approx \sqrt{n}$)
- [ ] Conformal wrapper for calibrated intervals
- [ ] Benchmark against SmallMLP, GPy, scikit-learn GP
- [ ] PyPI release

See [CHANGELOG.md](CHANGELOG.md) for progress.

---

## Planned API

```python
import numpy as np
from smallgp import SmallGPRegressor

X = np.random.randn(200, 5)
y = np.sin(X[:, 0]) + 0.1 * np.random.randn(200)

model = SmallGPRegressor()          # adaptive kernel, exact inference
model.fit(X, y)
y_hat = model.predict(X)
lo, hi = model.predict_interval(X, alpha=0.1)
```

Sparse variant:

```python
model = SmallGPRegressor(sparse=True)   # inducing points, n > 500
```

---

## Planned Method

### 1. Adaptive lengthscale

Kernel lengthscale initialized from data statistics:

$$
\ell_j = \text{std}(X_{:,j}) \cdot \sqrt{\frac{d}{n}}
$$

with per-feature ARD refinement via marginal likelihood on a validation split.

### 2. Adaptive noise

Observation noise initialized as a fraction of target variance:

$$
\sigma_n^2 = 0.01 \cdot \text{Var}(y)
$$

### 3. Sparse approximation

For $n > 500$, use inducing points at a subset of training data:

$$
m = \min\left(\left\lfloor \sqrt{n} \cdot \log_2 d \right\rfloor, 256\right)
$$

Optimize inducing point locations via marginal likelihood.

### 4. Conformal wrapper

Same weighted conformal approach as SmallMLP: locally adaptive intervals via an embedding kernel.

---

## When to use SmallGP (planned)

**Good fit:**
- Small datasets ($n < 500$) with **smooth** nonlinear structure.
- Scientific instruments: sensors, spectroscopy, chemistry.
- When **calibrated uncertainty** matters.
- When you want **theory-backed** predictions (GP posterior).

**Not a good fit:**
- Non-smooth functions — GPs assume smoothness.
- Very high dimensions ($d > 100$) with tiny $n$.
- Large datasets ($n > 10{,}000$) — use sparse GP or neural networks.

---

## Roadmap

| Milestone | Status |
|-----------|--------|
| Adaptive lengthscale | planned |
| Exact GP baseline | planned |
| Sparse GP | planned |
| Conformal wrapper | planned |
| Benchmark (45 datasets) | planned |
| PyPI release | planned |
| arXiv preprint | planned |

---

## Related work

- **SmallGBM** — gradient boosting for small tabular data. [GitHub](https://github.com/nsdmlk/SmallGBM)
- **SmallMLP** — adaptive MLP for small nonlinear data. [GitHub](https://github.com/nsdmlk/SmallMLP)

Part of the **Small ML** series: tuning-free models for small data.

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgments

Built independently during undergraduate studies at Beijing Institute of Technology.
