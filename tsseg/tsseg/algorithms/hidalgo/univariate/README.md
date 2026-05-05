# Hidalgo univariate via time-delay embedding — abandoned track

> **Status: ARCHIVED** — experiment concluded negatively.
> This code is retained for documentation purposes.

---

## Motivation

Hidalgo segments multivariate data into regions of differing **intrinsic
dimension** (ID). It does not natively support univariate time series,
because points of a scalar series lie on a 1-D curve ($d = 1$ everywhere).

The idea was to use a **Takens time-delay embedding** to lift the
univariate signal into $\mathbb{R}^m$:

$$X_t = (x_t,\; x_{t+\tau},\; x_{t+2\tau},\; \dots,\; x_{t+(m-1)\tau})$$

In embedding space, different dynamical regimes (periodic, chaotic,
noise) may occupy manifolds of different dimension, making Hidalgo
applicable.

## Implementation

`embedding_detector.py` contains `HidalgoEmbeddingDetector`, a wrapper
around `HidalgoDetector` that:

1. Applies the time-delay embedding ($m$, $\tau$) to univariate series
2. Feeds the embedded data to Hidalgo
3. Maps labels back to the original time axis (padding by the last label)

### Additional parameters

| Parameter    | Default | Description |
|--------------|---------|-------------|
| `embed_dim`  | 10      | Embedding dimension $m$ |
| `delay`      | 1       | Time delay $\tau$ |

Other parameters are inherited from `HidalgoDetector` (see the main
README).

---

## Experimental protocol

### Benchmarks

- **TSSB** (Time Series Segmentation Benchmark): 75 univariate series
    built by concatenating z-normalized UCR patterns.
- **UTSA**: 32 univariate series.

### Mode

Semi-supervised: $K$ set from ground truth (number of states).

### Configurations tested

- `embed_dim` ∈ {3, 5, 10}
- `delay` ∈ {1, 2}
- `n_iter` ∈ {300, 500, 1000}
- `q` = 3, `zeta` = 0.8 (values recommended by the paper)

---

## Results

### TSSB results (8 series tested, embed_dim=10, n_iter=300–500)

| Series | T | K | ARI | $\hat{d}$ per state |
|--------|---|---|-----|---------------------|
| 67 (ToeSegmentation2) | 471  | 2 | −0.002 | [4.96, 3.86] |
| 21 (ECGFiveDays)      | 782  | 2 | 0.008  | [3.91, 4.25] |
| 6 (CBF)               | 960  | 3 | 0.026  | [8.48, 6.97, 7.56] |
| 10 (Coffee)           | 1000 | 2 | −0.001 | [3.32, 3.28] |
| 38 (Lightning7)       | 1194 | 2 | 0.010  | [6.42, 5.65] |
| 35 (LargeKitchenAppliances) | 1280 | 4 | 0.016 | [4.33, 5.83, 5.60, 6.08] |
| 16 (CricketZ)         | 1293 | 3 | −0.001 | [3.40, 3.49, 3.86] |
| 58 (SonyAIBORobotSurface2) | 1400 | 2 | 0.021 | [6.62, 6.37] |

**Mean ARI: 0.009** (≈ random).

### Coffee series (tsseg-exp pipeline)

| embed_dim | ARI    | state_matching |
|-----------|--------|----------------|
| 3         | −0.001 | 0.390          |
| 10        | −0.001 | 0.389          |

### Favorable synthetic case (sine vs white noise, T=2000)

| $m$ | $\tau$ | $\hat{d}$         | ARI   |
|-----|--------|-------------------|-------|
| 3   | 1      | [2.32, 1.93]      | 0.005 |
| 5   | 2      | [2.09, 3.55]      | 0.124 |
| 10  | 1      | [8.31, 2.37]      | **0.571** |
| 10  | 2      | [8.95, 2.45]      | **0.546** |

The only working case is a synthetic signal with a **true change in
intrinsic dimension** (sine ≈ 1D vs white noise ≈ 10D).

---

## Intrinsic-dimension analysis per segment (oracle TWO-NN)

To understand the failure modes, we estimated the ID per ground-truth
segment using the TWO-NN estimator in the embedding space ($m = 10$,
$\tau = 1$).

### TSSB (69 series analyzed)

| Statistic | Value |
|-----------|-------|
| Median ID gap between segments | 1.19 |
| Fraction with gap > 1.0        | 56% |
| Fraction with gap > 2.0        | 23% |
| Max gap                        | 6.22 |

### UTSA (32 series analyzed)

| Statistic | Value |
|-----------|-------|
| Median ID gap between segments | 0.72 |
| Fraction with gap > 1.0        | 34% |
| Max gap                        | 6.49 |

---

## Diagnosis: three failure causes

### 1. CPD benchmarks do not test geometric-complexity changes

TSSB and UTSA are constructed by concatenating z-normalized UCR patterns.
Segments differ by signal *morphology* (sine vs triangle, different ECG
classes), not by the *intrinsic dimensionality* of the manifold. Two
different waveform shapes after delay embedding often remain ~1D curves
in $\mathbb{R}^m$ → identical ID.

Statistics confirm this: segments are z-normalized per segment (mean ≈ 0,
std ≈ 1 everywhere).

### 2. Hidalgo does not discriminate small ID gaps

Even when TWO-NN detects a gap (e.g. 3.07 vs 3.52), Hidalgo's Gibbs
sampler returns neighboring $\hat{d}$ values and largely uniform labels
across segments. The Potts term enforces local homogeneity, but it cannot
create a discriminative signal where none exists.

### 3. Computational cost

- $O(T^2)$ for NN distance computations
- $O(T \cdot K \cdot n_{\text{iter}})$ for Gibbs sampling
- ~20–35s per series for T ≈ 1000
- Full TSSB benchmark: only 2 trials completed in ~4h of execution

---

## Conclusion

**Hidalgo with time-delay embedding is not suitable for segmentation of
univariate series** on standard CPD benchmarks (TSSB, UTSA).

The reason is intrinsic: Hidalgo detects changes in *intrinsic
dimension* (geometric complexity), while CPD benchmarks contain changes
in *waveform shape/statistics*. This is not a matter of hyperparameter
tuning.

Hidalgo remains appropriate for its original use case: **multivariate**
data where segments lie on submanifolds of different dimensions.

The only viable 1D scenario is a signal that exhibits genuine changes in
dynamical regime (periodic → chaotic), which does not match standard
CPD benchmarks.
