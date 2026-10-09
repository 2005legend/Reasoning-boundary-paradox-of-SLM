"""GRPO advantage-gating strategies.

  VanillaGate    (Req 6)
  CBGRPOGate     (macro; Req 7, spend = max(advantage, 0))
  SELFGate       (per-prompt; SELF selection of arXiv:2510.02230: skip prompts whose greedy
                  answer is already correct. Gate id 'o_self' kept for spec continuity.)
  MEGGate        (micro; rarity credit redistribution over correct rollouts, the rule of
                  Cue-GRPO arXiv:2608.03467, applied to MEG's solution-mode partition)
  BBGGate        (per-prompt; GRPO-compatible variant of Bayesian Boundary Gating, arXiv:2606.15455)
  CompositeGate  (macro x micro -> h_cb_grpo or heg_grpo)

Weights are non-negative multipliers on the GRPO advantage. SELF and BBG only scale down; MEG
redistributes (mean 1 over a group's correct rollouts) and CB, by default, redistributes positive
credit across topics (mean 1 over the step's positive rollouts), so individual weights can exceed 1.
State updates go through local_increments / apply_synced_increments so the code stays DDP-safe.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .clustering import get_sentence_model
from .config import EMBED_MODEL, GateConfig
from .evaluation import gini_coefficient


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


SyncFn = Callable[[np.ndarray], np.ndarray]


def identity_sync(x: np.ndarray) -> np.ndarray:
    return x  # no-op for single-process / unit tests


@dataclass
class GateContext:
    """Everything a gate may look at for one micro-step. Rollouts of a prompt are contiguous."""
    cluster_ids: np.ndarray                  # macro topic id per rollout
    prompt_ids: np.ndarray
    is_correct: np.ndarray                   # 0/1 per rollout
    group_size: int
    partition: Optional[np.ndarray] = None   # solution-mode label per rollout; -1 for incorrect
    greedy_correct: Optional[np.ndarray] = None  # per rollout: is its prompt's greedy answer correct

    @staticmethod
    def build(cluster_ids, prompt_ids, is_correct, group_size, partition=None, greedy_correct=None):
        return GateContext(np.asarray(cluster_ids, dtype=np.int64), np.asarray(prompt_ids, dtype=np.int64),
                           np.asarray(is_correct, dtype=np.float64), int(group_size),
                           None if partition is None else np.asarray(partition, dtype=np.int64),
                           None if greedy_correct is None else np.asarray(greedy_correct, dtype=bool))


class Gate(ABC):
    name: str = 'abstract'
    requires_partition: bool = False
    requires_greedy: bool = False

    @abstractmethod
    def compute_weights(self, ctx: GateContext) -> np.ndarray: ...

    def local_increments(self, ctx: GateContext, advantages) -> dict:
        return {}

    def apply_synced_increments(self, synced: dict) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, sd: dict) -> None:
        pass

    def step(self, ctx: GateContext, advantages, sync_fn: SyncFn = identity_sync) -> np.ndarray:
        """Weights from the current (pre-update) state, then update state with synced increments."""
        weights = self.compute_weights(ctx)
        synced = {k: sync_fn(v) for k, v in self.local_increments(ctx, advantages).items()}
        self.apply_synced_increments(synced)
        return weights


class VanillaGate(Gate):
    name = 'vanilla'

    def compute_weights(self, ctx: GateContext) -> np.ndarray:
        return np.ones(len(ctx.cluster_ids), dtype=np.float64)


# ── CBGRPOGate (macro) ───────────────────────────────────────────────────────

class CBGRPOGate(Gate):
    """Topic budget: topics whose EMA of positive advantage mass ('spend') exceeds theta x mean get
    decay ** (ratio - theta) of the positive credit.

    mean_preserving=True (pre-registered 2026-10-06): the budget only touches positive-advantage
    rollouts (correct ones in mixed groups), and their weights are rescaled to mean 1 over the step,
    so credit moves from over-spending topics to the rest instead of the positive learning rate
    dropping (the confound MEG's reformulation removed). Negative advantages are never throttled
    (Req 39.2). mean_preserving=False is the original rule: down-weight only, every rollout of the topic."""
    name = 'cb_grpo'

    def __init__(self, n_clusters: int, theta: float = 1.5, decay: float = 0.98,
                 ema_alpha: float = 0.05, use_positive_mass_only: bool = True,
                 mean_preserving: bool = False, clip_min: float = 0.3, clip_max: float = 3.0):
        self.n_clusters = n_clusters
        self.theta = theta
        self.decay = decay
        self.ema_alpha = ema_alpha
        self.use_positive_mass_only = use_positive_mass_only
        self.mean_preserving = mean_preserving
        self.clip_min, self.clip_max = clip_min, clip_max
        self.spend_ema = np.zeros(n_clusters, dtype=np.float64)

    def compute_weights(self, ctx: GateContext) -> np.ndarray:
        ratio = self.spend_ema[ctx.cluster_ids] / (self.spend_ema.mean() + 1e-8)
        over = np.clip(ratio - self.theta, a_min=0.0, a_max=None)
        raw = np.where(ratio > self.theta, self.decay ** over, 1.0)
        if not self.mean_preserving:
            return raw
        # positive advantage <=> correct rollout in a group that is neither all-right nor all-wrong
        n_correct = ctx.is_correct.reshape(-1, ctx.group_size).sum(axis=1)
        mixed = np.repeat((n_correct > 0) & (n_correct < ctx.group_size), ctx.group_size)
        pos = mixed & (ctx.is_correct > 0)
        w = np.ones(len(raw), dtype=np.float64)
        if pos.any():
            p = raw[pos] / raw[pos].mean()
            p = np.clip(p, self.clip_min, self.clip_max)
            w[pos] = p / p.mean()
        return w

    def local_increments(self, ctx: GateContext, advantages) -> dict:
        adv = _to_numpy(advantages)
        mass = np.maximum(adv, 0.0) if self.use_positive_mass_only else np.abs(adv)
        sum_ = np.zeros(self.n_clusters, dtype=np.float64)
        count = np.zeros(self.n_clusters, dtype=np.float64)
        np.add.at(sum_, ctx.cluster_ids, mass)
        np.add.at(count, ctx.cluster_ids, 1.0)
        return {'cb_mass_sum': sum_, 'cb_mass_count': count}

    def apply_synced_increments(self, synced: dict) -> None:
        s, c = synced['cb_mass_sum'], synced['cb_mass_count']
        touched = c > 0
        mean_step = np.zeros(self.n_clusters, dtype=np.float64)
        mean_step[touched] = s[touched] / c[touched]
        self.spend_ema[touched] = (self.ema_alpha * mean_step[touched]
                                   + (1 - self.ema_alpha) * self.spend_ema[touched])

    def state_dict(self) -> dict:
        return {'spend_ema': self.spend_ema.copy()}

    def load_state_dict(self, sd: dict) -> None:
        self.spend_ema = np.asarray(sd['spend_ema'], dtype=np.float64).copy()

    def diagnostic_snapshot(self) -> dict:
        return {'cluster_spend_ema': self.spend_ema.tolist(), 'gini': gini_coefficient(self.spend_ema)}


# ── SELFGate (per-prompt, the base paper's selection rule) ───────────────────

class SELFGate(Gate):
    """SELF selection (arXiv:2510.02230 Sec. 6): learn only on problems whose greedy answer fails.
    Prompts whose greedy answer is correct get weight self_lambda (0 = the paper's hard filter).
    Evaluated fresh every step from that step's greedy answer, so it is stateless and active from
    step 0 (the earlier EMA variant never reached its threshold at ~2 visits per prompt).
    Only the selection is implemented, not SELF's forward-KL term."""
    name = 'o_self'
    requires_greedy = True

    def __init__(self, self_lambda: float = 0.0):
        self.self_lambda = self_lambda

    def compute_weights(self, ctx: GateContext) -> np.ndarray:
        if ctx.greedy_correct is None:
            return np.ones(len(ctx.cluster_ids), dtype=np.float64)
        return np.where(ctx.greedy_correct, self.self_lambda, 1.0)


# ── Solution-mode partitions (feed MEGGate and the diagnostics of every condition) ──

_WORD_RE = re.compile(r'\w+|[^\w\s]')


def _bigram_set(text: str) -> set:
    toks = _WORD_RE.findall(text.lower())
    return set(zip(toks, toks[1:])) or {tuple(toks)}


def _bigram_distance_matrix(texts: Sequence[str]) -> np.ndarray:
    sets = [_bigram_set(t) for t in texts]
    n = len(sets)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            union = len(sets[i] | sets[j])
            d[i, j] = d[j, i] = 1.0 - (len(sets[i] & sets[j]) / union if union else 1.0)
    return d


def _agglomerative_labels(dist: np.ndarray, threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    if len(dist) < 2:
        return np.zeros(len(dist), dtype=np.int64)
    return AgglomerativeClustering(n_clusters=None, distance_threshold=max(threshold, 1e-6),
                                   metric='precomputed', linkage='average').fit(dist).labels_.astype(np.int64)


def _real_partition(texts: Sequence[str], method: str, tau_mode: float, bigram_threshold: float,
                    embeddings: Optional[np.ndarray]) -> np.ndarray:
    if method == 'bigram':
        return _agglomerative_labels(_bigram_distance_matrix(texts), bigram_threshold)
    sims = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    return _agglomerative_labels(1.0 - sims, 1.0 - tau_mode)


def compute_group_partitions(completions: Sequence[str], is_correct, group_size: int,
                             method: str = 'embedding', tau_mode: float = 0.85,
                             bigram_threshold: float = 0.5, embedding_model: str = EMBED_MODEL,
                             device: str = 'cpu') -> tuple:
    """Cluster each group's CORRECT rollouts into solution modes.

    method: 'embedding' (MiniLM cosine, agglomerative at 1 - tau_mode) or 'bigram' (word-bigram
    Jaccard distance, agglomerative at bigram_threshold; arXiv:2606.29985 finds bigram distance
    tracks approach-level differences better than embedding cosine).

    Returns (labels, diagnostics): labels[i] >= 0 for correct rollouts, -1 for incorrect.
    Diagnostics, over groups with >= 2 correct rollouts: mean number of modes, mean normalized
    mode entropy (0 = every correct rollout in one mode, 1 = all distinct), fraction multi-mode.
    """
    if method not in ('embedding', 'bigram'):
        raise ValueError(f'partition method must be embedding or bigram, got {method!r}')
    correct = np.asarray(is_correct, dtype=bool)
    n = len(completions)
    labels = np.full(n, -1, dtype=np.int64)
    real_method = method
    embeddings = None
    if real_method == 'embedding' and correct.sum() >= 2:
        idx = np.flatnonzero(correct)
        emb = get_sentence_model(embedding_model, device).encode(
            [completions[i] for i in idx], batch_size=128, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.zeros((n, emb.shape[1]), dtype=np.float32)
        embeddings[idx] = emb
    n_modes, entropies = [], []
    for start in range(0, n, group_size):
        idx = np.flatnonzero(correct[start:start + group_size]) + start
        if len(idx) == 0:
            continue
        if len(idx) == 1:
            labels[idx] = 0
            continue
        texts = [completions[i] for i in idx]
        lab = _real_partition(texts, real_method, tau_mode, bigram_threshold,
                              None if embeddings is None else embeddings[idx])
        labels[idx] = lab
        counts = np.bincount(lab)
        counts = counts[counts > 0]
        p = counts / counts.sum()
        n_modes.append(len(counts))
        entropies.append(float(-(p * np.log(p)).sum() / np.log(len(idx))))
    diag = {}
    if n_modes:
        diag = {'mean_correct_modes': float(np.mean(n_modes)),
                'mean_group_mode_entropy': float(np.mean(entropies)),
                'frac_groups_multi_mode': float(np.mean(np.asarray(n_modes) > 1))}
    return labels, diag


def randomize_partition(labels: np.ndarray, group_size: int, rng: np.random.Generator) -> np.ndarray:
    """Random-partition control (Cue-GRPO's Random-Cluster): within each group, correct rollouts get
    labels drawn uniformly over as many clusters as the real partition found; incorrect stay -1."""
    out = np.array(labels, dtype=np.int64, copy=True)
    for start in range(0, len(out), group_size):
        idx = np.flatnonzero(out[start:start + group_size] >= 0) + start
        if len(idx) >= 2:
            out[idx] = rng.integers(0, len(np.unique(out[idx])), size=len(idx))
    return out


# ── MEGGate (micro): rarity credit redistribution ────────────────────────────

class MEGGate(Gate):
    """Redistributes positive credit among a group's correct rollouts by mode rarity
    (Cue-GRPO arXiv:2608.03467, Eq. 3, mean-matched variant):
        w_i = N |C_i|^-alpha / sum_j |C_j|^-alpha  over the N correct rollouts, clipped, then
        rescaled to mean 1;  incorrect rollouts keep weight 1.
    Mean 1 means MEG changes WHO gets credit, not how much credit there is in total: this removes
    the earlier confound where a down-weight-only gate also acted as a lower learning rate."""
    name = 'meg'
    requires_partition = True

    def __init__(self, alpha: float = 0.8, clip_min: float = 0.3, clip_max: float = 3.0):
        self.alpha = alpha
        self.clip_min = clip_min
        self.clip_max = clip_max

    def compute_weights(self, ctx: GateContext) -> np.ndarray:
        n = len(ctx.cluster_ids)
        weights = np.ones(n, dtype=np.float64)
        if ctx.partition is None:
            return weights
        for start in range(0, n, ctx.group_size):
            block = ctx.partition[start:start + ctx.group_size]
            idx = np.flatnonzero(block >= 0) + start
            if len(idx) < 2:
                continue
            lab = ctx.partition[idx]
            _, inverse, counts = np.unique(lab, return_inverse=True, return_counts=True)
            s = counts[inverse].astype(np.float64) ** (-self.alpha)
            w = np.clip(len(idx) * s / s.sum(), self.clip_min, self.clip_max)
            weights[idx] = w * len(idx) / w.sum()
        return weights


# ── BBGGate (per-prompt): GRPO-compatible Bayesian Boundary Gating ───────────

def bbg_utility(m: int, n: int, k: int) -> float:
    """u(m; n, k) = E_{p ~ Beta(m+1, n-m+1)}[k (1-p)^(k-1)] = k prod_{j=1}^{k-1} (n-m+j)/(n+1+j)
    (arXiv:2606.15455, Eq. 8)."""
    out = float(k)
    for j in range(1, k):
        out *= (n - m + j) / (n + 1 + j)
    return out


class BBGGate(Gate):
    """Problem-level weight from the group's success count m: u(m)/u(1), hard-gated to 0 when
    u(m) < tau (n+1) (the paper's Eq. 9 threshold). Groups with m = 0 or m = n have zero GRPO
    advantage anyway. NOT a reproduction: BBG's m = 0 bucket learns through a signed REINFORCE
    loss, which group-normalized GRPO cannot express."""
    name = 'bbg'

    def __init__(self, k_ref: int = 16, tau: float = 0.01):
        self.k_ref = k_ref
        self.tau = tau

    def bucket_weights(self, n: int) -> np.ndarray:
        u = np.array([bbg_utility(m, n, self.k_ref) for m in range(n + 1)])
        w = np.where(u >= self.tau * (n + 1), u / u[1], 0.0)
        w[0] = 1.0  # irrelevant (zero advantage); keeps the logged mean weight interpretable
        return w

    def compute_weights(self, ctx: GateContext) -> np.ndarray:
        n_rollouts = len(ctx.cluster_ids)
        table = self.bucket_weights(ctx.group_size)
        weights = np.ones(n_rollouts, dtype=np.float64)
        for start in range(0, n_rollouts, ctx.group_size):
            m = int(round(ctx.is_correct[start:start + ctx.group_size].sum()))
            weights[start:start + ctx.group_size] = table[m]
        return weights


# ── CompositeGate (macro x micro) ────────────────────────────────────────────

def _combine(macro_w: np.ndarray, micro_w: np.ndarray, op: str) -> np.ndarray:
    """Req 39.5: configurable combination operator."""
    if op == 'multiplicative':
        return macro_w * micro_w
    if op == 'min':
        return np.minimum(macro_w, micro_w)
    if op == 'harmonic_mean':
        return 2 * macro_w * micro_w / (macro_w + micro_w + 1e-8)
    raise ValueError(f'Unknown combine op: {op}')


class CompositeGate(Gate):
    """macro=CBGRPOGate with micro=SELFGate -> 'h_cb_grpo'; with micro=MEGGate -> 'heg_grpo'."""

    def __init__(self, macro: Gate, micro: Gate, combine_op: str = 'multiplicative'):
        self.macro = macro
        self.micro = micro
        self.combine_op = combine_op
        self.name = 'heg_grpo' if isinstance(micro, MEGGate) else 'h_cb_grpo'

    @property
    def requires_partition(self) -> bool:
        return self.macro.requires_partition or self.micro.requires_partition

    @property
    def requires_greedy(self) -> bool:
        return self.macro.requires_greedy or self.micro.requires_greedy

    def compute_weights(self, ctx: GateContext) -> np.ndarray:
        return _combine(self.macro.compute_weights(ctx), self.micro.compute_weights(ctx), self.combine_op)

    def local_increments(self, ctx: GateContext, advantages) -> dict:
        merged = {f'macro__{k}': v for k, v in self.macro.local_increments(ctx, advantages).items()}
        merged.update({f'micro__{k}': v for k, v in self.micro.local_increments(ctx, advantages).items()})
        return merged

    def apply_synced_increments(self, synced: dict) -> None:
        self.macro.apply_synced_increments({k[7:]: v for k, v in synced.items() if k.startswith('macro__')})
        self.micro.apply_synced_increments({k[7:]: v for k, v in synced.items() if k.startswith('micro__')})

    def state_dict(self) -> dict:
        return {'macro': self.macro.state_dict(), 'micro': self.micro.state_dict(),
                'combine_op': self.combine_op}

    def load_state_dict(self, sd: dict) -> None:
        self.macro.load_state_dict(sd['macro'])
        self.micro.load_state_dict(sd['micro'])
        self.combine_op = sd.get('combine_op', self.combine_op)

    def diagnostic_snapshot(self) -> dict:
        return self.macro.diagnostic_snapshot() if hasattr(self.macro, 'diagnostic_snapshot') else {}


def build_gate(gate_type: str, n_clusters: int, cfg: GateConfig) -> Gate:
    def cb():
        return CBGRPOGate(n_clusters, theta=cfg.cb_theta, decay=cfg.cb_decay, ema_alpha=cfg.cb_ema_alpha,
                          use_positive_mass_only=cfg.cb_use_positive_mass_only,
                          mean_preserving=cfg.cb_mean_preserving, clip_min=cfg.cb_clip_min,
                          clip_max=cfg.cb_clip_max)

    def meg():
        return MEGGate(alpha=cfg.meg_alpha, clip_min=cfg.meg_clip_min, clip_max=cfg.meg_clip_max)

    if gate_type == 'vanilla':
        return VanillaGate()
    if gate_type == 'cb_grpo':
        return cb()
    if gate_type == 'o_self':
        return SELFGate(cfg.self_lambda)
    if gate_type == 'meg':
        return meg()
    if gate_type == 'h_cb_grpo':
        return CompositeGate(cb(), SELFGate(cfg.self_lambda), combine_op=cfg.combine_op)
    if gate_type == 'heg_grpo':
        return CompositeGate(cb(), meg(), combine_op=cfg.combine_op)
    if gate_type == 'bbg':
        return BBGGate(k_ref=cfg.bbg_k_ref, tau=cfg.bbg_tau)
    raise ValueError(f'Unknown gate_type: {gate_type}')
