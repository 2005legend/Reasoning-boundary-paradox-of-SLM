"""Pass@k, shrinkage slope, multi-seed significance testing (Req 15, 16, 41, 42)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.special import comb


def pass_at_k_unbiased(num_samples: int, num_correct: int, k: int) -> float:
    """Req 15.4: Chen et al. (2021) unbiased estimator."""
    n, c = num_samples, num_correct
    if k > n:
        raise ValueError(f'k={k} cannot exceed n={n}.')
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k, exact=True) / comb(n, k, exact=True)


def compute_pass_at_k(correctness_matrix: np.ndarray, k_values: Sequence) -> dict:
    """correctness_matrix: [num_problems, n] of 0/1. Returns {k: mean Pass@k}."""
    n = correctness_matrix.shape[1]
    num_correct = correctness_matrix.sum(axis=1).astype(int)
    return {int(k): float(np.mean([pass_at_k_unbiased(n, int(c), k) for c in num_correct]))
            for k in k_values}


def check_ceiling_effect(pass_at_k: dict, threshold: float = 0.90) -> dict:
    """Req 41.2: flag k-values where base Pass@k >= threshold."""
    return {k: (v >= threshold) for k, v in pass_at_k.items()}


@dataclass
class ShrinkageSlopeResult:
    slope: float
    intercept: float
    r_squared: float
    ci_lower: float
    ci_upper: float
    delta_pass_k: dict = field(default_factory=dict)


def compute_shrinkage_slope(base_pass_k: dict, rl_pass_k: dict,
                            n_bootstrap: int = 1000, seed: int = 0) -> ShrinkageSlopeResult:
    """Req 16: regress delta Pass@k on log(k), with bootstrap 95% CI over k-points."""
    base = {int(k): v for k, v in base_pass_k.items()}
    rl = {int(k): v for k, v in rl_pass_k.items()}
    k_values = sorted(set(base) & set(rl))
    if len(k_values) < 2:
        raise ValueError('Need at least 2 shared k-values to fit a slope.')
    delta = np.array([rl[k] - base[k] for k in k_values])
    log_k = np.log(np.array(k_values, dtype=np.float64))
    slope, intercept = np.polyfit(log_k, delta, deg=1)
    predicted = slope * log_k + intercept
    ss_res = np.sum((delta - predicted) ** 2)
    ss_tot = np.sum((delta - delta.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    rng = np.random.default_rng(seed)
    idx_pool = np.arange(len(k_values))
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(idx_pool, size=len(idx_pool), replace=True)
        if np.ptp(log_k[idx]) == 0:
            continue
        boot.append(np.polyfit(log_k[idx], delta[idx], deg=1)[0])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5]) if boot else (float('nan'), float('nan'))
    return ShrinkageSlopeResult(slope=float(slope), intercept=float(intercept),
                                r_squared=float(r_squared), ci_lower=float(ci_lo), ci_upper=float(ci_hi),
                                delta_pass_k={k: float(d) for k, d in zip(k_values, delta)})


@dataclass
class DeltaSlopeTestResult:
    mean_delta: float
    ci_lower: float
    ci_upper: float
    excludes_zero: bool
    n_seeds: int


def bootstrap_delta_slope_ci(condition_slopes: Sequence, baseline_slopes: Sequence,
                             n_bootstrap: int = 5000, seed: int = 0) -> DeltaSlopeTestResult:
    """Req 42.3-42.4: bootstrap CI on mean(condition) - mean(baseline) across seeds."""
    cond = np.asarray(condition_slopes, dtype=np.float64)
    base = np.asarray(baseline_slopes, dtype=np.float64)
    if min(len(cond), len(base)) < 2:
        raise ValueError('Need at least 2 seeds per condition for a meaningful CI (Req 42.1).')
    rng = np.random.default_rng(seed)
    diffs = np.array([cond[rng.choice(len(cond), size=len(cond), replace=True)].mean()
                      - base[rng.choice(len(base), size=len(base), replace=True)].mean()
                      for _ in range(n_bootstrap)])
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return DeltaSlopeTestResult(mean_delta=float(cond.mean() - base.mean()),
                                ci_lower=float(ci_lo), ci_upper=float(ci_hi),
                                excludes_zero=bool(ci_lo > 0 or ci_hi < 0),
                                n_seeds=min(len(cond), len(base)))


def gini_coefficient(values: np.ndarray) -> float:
    """Req 24.4 / 40.6: cluster-spend imbalance."""
    x = np.asarray(values, dtype=np.float64)
    if np.any(x < 0):
        x = x - x.min()
    if x.sum() == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cum = np.cumsum(sorted_x)
    return float((2 * np.sum(np.arange(1, n + 1) * sorted_x) - (n + 1) * cum[-1]) / (n * cum[-1]))
