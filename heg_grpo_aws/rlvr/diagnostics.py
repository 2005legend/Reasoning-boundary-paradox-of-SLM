"""Mechanism-level diagnostics (Req 40 / N5). Run on saved checkpoints; no training cost."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class InterferenceSnapshot:
    delta_plus: float  # mean(log pi_new(y+|x) - log pi_old(y+|x))  [SELF paper Def. 4.1]
    delta_sq: float    # mean((log pi_new - log pi_old)^2)


def compute_interference(logprob_before: np.ndarray, logprob_after: np.ndarray) -> InterferenceSnapshot:
    before = np.asarray(logprob_before, dtype=np.float64)
    after = np.asarray(logprob_after, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError('before/after arrays must have the same shape (same probing set).')
    delta = after - before
    return InterferenceSnapshot(delta_plus=float(delta.mean()), delta_sq=float(np.mean(delta ** 2)))


def entropy_from_probs(probs: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def mean_token_entropy(token_entropies: Sequence) -> float:
    per_seq = [float(np.mean(t)) for t in token_entropies if len(t) > 0]
    return float(np.mean(per_seq)) if per_seq else float('nan')


@dataclass
class CorrelationResult:
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n: int
    note: str


def correlate_exploratory(x: Sequence, y: Sequence, label: str = '') -> CorrelationResult:
    """Req 40.6-40.7: descriptive only at this n; never report as a hypothesis test."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')
    if n < 3:
        pr = pp = sr = sp = float('nan')
    else:
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
    suffix = f': {label}' if label else ''
    detail = ('Too few points for any meaningful correlation; report the scatter, not a coefficient.'
              if n < 5 else 'Exploratory/descriptive only; not a hypothesis test at this n.')
    return CorrelationResult(pearson_r=float(pr), pearson_p=float(pp), spearman_r=float(sr),
                             spearman_p=float(sp), n=n, note=f'n={n} conditions{suffix}. {detail}')
