
# Core imports used throughout the notebook
from __future__ import annotations

import json, os, random, re, time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.special import comb
from sklearn.cluster import KMeans
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)

print('✅ Core imports OK')
# ── Standalone-script path roots (not from notebook globals) ───────────────
ARTIFACT_ROOT = Path('/kaggle/working/artifacts')
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# config.py — Centralized experiment configuration
# Implements: Req 2, Req 33, Req 39 (H-CB-GRPO gate type), Req 41-43
# ════════════════════════════════════════════════════════════════════════════

ModelSize = Literal['0.5B', '1.5B']
GateType = Literal[
    'vanilla',
    'cb_grpo',       # macro-only, refined (positive-mass spend)  [Req 7, Req 39.2]
    'o_self',        # micro-only, per-prompt solve-rate gate      [Req 8]
    'h_cb_grpo',     # CompositeGate: macro × micro                [Req 39 / N4]
]
ComputeTier = Literal['smoke', 'standard', 'final']
RewardMode = Literal['positive_only', 'negative_only', 'hybrid']
CombineOp = Literal['multiplicative', 'min', 'harmonic_mean']

MODEL_NAME_BY_SIZE = {
    '0.5B': 'Qwen/Qwen2.5-0.5B-Instruct',
    '1.5B': 'Qwen/Qwen2.5-1.5B-Instruct',
}

# Req 33.4 + Req 39.8: gates that need per-prompt state are memory-prohibitive at 0.5B
INCOMPATIBLE_GATES_AT_0_5B = {'o_self', 'h_cb_grpo'}

# (training_steps, checkpoint_interval, eval_interval, log_interval)
TIER_SETTINGS: dict[str, dict[str, int]] = {
    'smoke':    {'training_steps': 80,   'checkpoint_interval': 40,  'eval_interval': 40,  'log_interval': 5},
    'standard': {'training_steps': 800,  'checkpoint_interval': 50,  'eval_interval': 150, 'log_interval': 5},
    'final':    {'training_steps': 1800, 'checkpoint_interval': 150, 'eval_interval': 200, 'log_interval': 10},
}


@dataclass
class ClusterConfig:
    n_clusters: int = 16
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    basis: Literal['semantic', 'difficulty_aware'] = 'semantic'  # Req 43
    cache_dir: str = str(ARTIFACT_ROOT / 'clusters')
    random_state: int = 0


@dataclass
class GateConfig:
    gate_type: str = 'vanilla'
    # CB-GRPO (macro) — Req 7, refined per addendum §4.1 (positive-mass spend)
    cb_theta: float = 1.5
    cb_decay: float = 0.98
    cb_ema_alpha: float = 0.05
    cb_use_positive_mass_only: bool = True  # Req 39.2: spend = max(adv, 0)
    # O-SELF (micro) — Req 8
    self_tau_solve: float = 0.7
    self_lambda: float = 0.3
    self_ema_alpha: float = 0.10
    # Composite (H-CB-GRPO) — Req 39.5
    combine_op: str = 'multiplicative'

    def validate(self, model_size: str) -> None:
        if self.gate_type in INCOMPATIBLE_GATES_AT_0_5B and model_size == '0.5B':
            raise ValueError(
                f"Gate '{self.gate_type}' is not supported at 0.5B "
                f"(Req 33.4-33.6 / Req 39.8). Use 1.5B or vanilla/cb_grpo."
            )
        if not (0.0 < self.self_lambda <= 1.0):
            raise ValueError('self_lambda must be in (0, 1].')
        if self.cb_theta <= 1.0:
            raise ValueError('cb_theta must be > 1.0 (over-budget threshold).')


@dataclass
class RewardConfig:
    reward_mode: str = 'positive_only'
    format_weight: float = 1.0
    correctness_weight: float = 1.0


@dataclass
class GenerationConfig:
    rollouts_per_prompt: int = 4           # G — Req 6.2
    eval_samples_per_problem: int = 30     # n for Pass@k — Req 41.3 (raised from 10)
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    eval_temperature: float = 0.6
    eval_top_p: float = 0.95
    gen_microbatch_prompts: int = 4        # VRAM safety knob for T4


@dataclass
class OptimConfig:
    learning_rate: float = 1e-6
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    kl_beta: float = 0.0
    num_ppo_epochs: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    per_device_prompts_per_step: int = 4
    grad_accum_steps: int = 4


@dataclass
class ExperimentConfig:
    exp_name: str = 'exp'
    model_size: str = '1.5B'
    compute_tier: str = 'standard'
    seed: int = 0

    gate: GateConfig = field(default_factory=GateConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    output_root: str = str(ARTIFACT_ROOT / 'runs')
    max_prompt_len: int = 512

    def __post_init__(self):
        self.gate.validate(self.model_size)

    @property
    def model_name(self) -> str:
        return MODEL_NAME_BY_SIZE[self.model_size]

    @property
    def training_steps(self) -> int:
        return TIER_SETTINGS[self.compute_tier]['training_steps']

    @property
    def checkpoint_interval(self) -> int:
        return TIER_SETTINGS[self.compute_tier]['checkpoint_interval']

    @property
    def eval_interval(self) -> int:
        return TIER_SETTINGS[self.compute_tier]['eval_interval']

    @property
    def log_interval(self) -> int:
        return TIER_SETTINGS[self.compute_tier]['log_interval']

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.exp_name / self.model_size / f'seed{self.seed}'

    def to_json(self, path: Optional[str] = None) -> str:
        text = json.dumps(asdict(self), indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    @staticmethod
    def from_json(path: str) -> 'ExperimentConfig':
        raw = json.loads(Path(path).read_text())
        raw['gate'] = GateConfig(**raw.pop('gate'))
        raw['reward'] = RewardConfig(**raw.pop('reward'))
        raw['generation'] = GenerationConfig(**raw.pop('generation'))
        raw['optim'] = OptimConfig(**raw.pop('optim'))
        raw['cluster'] = ClusterConfig(**raw.pop('cluster'))
        return ExperimentConfig(**raw)


print('✅ config module defined')# ════════════════════════════════════════════════════════════════════════════
# rewards.py — Output parsing, pretty-printing, reward computation
# Implements: Req 11, 12, 13, 36, 37, 38
# Runs entirely on CPU (regex + sympy). Never raises — returns None on failure.
# ════════════════════════════════════════════════════════════════════════════

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

_REASONING_RE = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL)
_ANSWER_RE    = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
_BOXED_RE     = re.compile(r'\\boxed\{(.*?)\}', re.DOTALL)


@dataclass
class ParsedOutput:
    reasoning: Optional[str]
    answer: Optional[str]
    boxed: Optional[str]

    @property
    def is_well_formed(self) -> bool:
        return self.reasoning is not None and self.answer is not None and self.boxed is not None


def _extract_outermost(pattern: re.Pattern, text: str) -> Optional[str]:
    """Req 36.5: for nested/duplicate tags, take the outermost span."""
    opening = pattern.pattern.split('(.*?)')[0]
    closing = pattern.pattern.split('(.*?)')[1]
    start = text.find(opening)
    if start == -1:
        return None
    end = text.rfind(closing)
    if end == -1 or end <= start:
        return None
    return text[start + len(opening): end].strip()


def parse_output(text: str) -> ParsedOutput:
    """Req 36: extract reasoning / answer / boxed value."""
    if text is None:
        return ParsedOutput(None, None, None)
    reasoning = _extract_outermost(_REASONING_RE, text)
    answer_block = _extract_outermost(_ANSWER_RE, text)
    boxed = None
    if answer_block is not None:
        m = _BOXED_RE.search(answer_block)
        if m:
            boxed = m.group(1).strip()
    return ParsedOutput(reasoning=reasoning, answer=answer_block, boxed=boxed)


def pretty_print(reasoning: str, answer: str, boxed_value: str) -> str:
    """Req 37: inverse of parse_output."""
    reasoning = (reasoning or '').strip()
    boxed_value = (boxed_value or '').strip()
    answer_body = (answer or '').strip()
    if f'\\boxed{{{boxed_value}}}' not in answer_body:
        answer_body = f'{answer_body}\n\\boxed{{{boxed_value}}}'.strip()
    return f'<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer_body}\n</answer>'


def format_reward(generated_text: str) -> float:
    """Req 11: 1.0 iff reasoning+answer+boxed all present."""
    return 1.0 if parse_output(generated_text).is_well_formed else 0.0


def _normalize_numeric(expr_str: str):
    """Req 12.2/12.3: sympy normalization with common latex cleanup."""
    if expr_str is None:
        return None
    cleaned = expr_str.strip()
    cleaned = cleaned.replace(',', '').replace('$', '').replace('\\!', '')
    cleaned = cleaned.rstrip('.')
    cleaned = cleaned.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    cleaned = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', cleaned)
    cleaned = cleaned.replace('\\%', '/100').replace('%', '/100')
    cleaned = cleaned.replace('^', '**')
    try:
        return parse_expr(cleaned, transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        return None


def answers_equivalent(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """Req 12.4: sympy equivalence. Returns False on any parsing failure."""
    if predicted is None or ground_truth is None:
        return False
    p = _normalize_numeric(predicted)
    g = _normalize_numeric(ground_truth)
    if p is None or g is None:
        return predicted.strip() == ground_truth.strip()
    try:
        diff = sympy.simplify(p - g)
        if diff == 0:
            return True
        return bool(abs(complex(diff.evalf())) < 1e-6)
    except Exception:
        return p == g


def correctness_reward(generated_text: str, ground_truth: str, reward_mode: str = 'positive_only') -> float:
    """Req 12: combine extraction + sympy equivalence + reward_mode."""
    parsed = parse_output(generated_text)
    is_correct = answers_equivalent(parsed.boxed, ground_truth)
    if reward_mode == 'positive_only': return 1.0 if is_correct else 0.0
    if reward_mode == 'negative_only': return 0.0 if is_correct else -1.0
    if reward_mode == 'hybrid':        return 1.0 if is_correct else -0.5
    raise ValueError(f'Unknown reward_mode: {reward_mode}')


@dataclass
class RewardBreakdown:
    format: float
    correctness: float
    total: float
    is_correct: bool
    is_well_formed: bool


def compute_reward(generated_text: str, ground_truth: str, reward_mode: str = 'positive_only',
                   format_weight: float = 1.0, correctness_weight: float = 1.0) -> RewardBreakdown:
    """Req 13: total reward = format_weight * format + correctness_weight * correctness."""
    parsed = parse_output(generated_text)
    f = format_reward(generated_text)
    c = correctness_reward(generated_text, ground_truth, reward_mode)
    return RewardBreakdown(
        format=f, correctness=c, total=format_weight * f + correctness_weight * c,
        is_correct=answers_equivalent(parsed.boxed, ground_truth),
        is_well_formed=parsed.is_well_formed
    )


print('✅ rewards module defined')# ════════════════════════════════════════════════════════════════════════════
# data.py — Dataset loading and prompt formatting (Req 4)
# Needs internet access on Kaggle (Settings → Internet → On)
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    'Please reason step by step. Wrap your reasoning in <reasoning></reasoning> tags, '
    'then give your final answer in <answer></answer> tags with the numeric result inside '
    '\\boxed{}.'
)


@dataclass
class Problem:
    prompt_id: str
    prompt_text: str       # raw question (for embedding/clustering)
    chat_prompt: str       # fully chat-templated, ready for tokenizer
    ground_truth: str


def _extract_gsm8k_answer(answer_field: str) -> str:
    """GSM8K's `answer` field ends with '#### <number>'."""
    match = re.search(r'####\s*([\-0-9,./]+)', answer_field)
    if not match:
        raise ValueError(f"Could not find '#### <answer>' in: {answer_field!r}")
    return match.group(1).replace(',', '').strip()


def format_chat_prompt(tokenizer, question: str) -> str:
    """Req 4.6: format with instruction template matching the base model's chat format."""
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': f"{question}\nLet's think step by step and put the final "
                                      f"numeric answer in \\boxed{{}}."}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_gsm8k(tokenizer, split: str = 'train') -> list:
    """Req 4.1/4.2: openai/gsm8k, main config."""
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=split)
    problems = []
    for i, row in enumerate(ds):
        gt = _extract_gsm8k_answer(row['answer'])
        problems.append(Problem(
            prompt_id=f'gsm8k_{split}_{i}',
            prompt_text=row['question'],
            chat_prompt=format_chat_prompt(tokenizer, row['question']),
            ground_truth=gt,
        ))
    return problems


def load_gsm8k_platinum(tokenizer, split: str = 'test') -> list:
    """Req 4.3: madrylab/gsm8k-platinum."""
    from datasets import load_dataset
    ds = load_dataset('madrylab/gsm8k-platinum', split=split)
    problems = []
    for i, row in enumerate(ds):
        gt = _extract_gsm8k_answer(row['answer'])
        problems.append(Problem(
            prompt_id=f'platinum_{split}_{i}',
            prompt_text=row['question'],
            chat_prompt=format_chat_prompt(tokenizer, row['question']),
            ground_truth=gt,
        ))
    return problems


def load_math500(tokenizer) -> list:
    """Req 4.4: HuggingFaceH4/MATH-500."""
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    problems = []
    for i, row in enumerate(ds):
        problems.append(Problem(
            prompt_id=f'math500_{i}',
            prompt_text=row['problem'],
            chat_prompt=format_chat_prompt(tokenizer, row['problem']),
            ground_truth=str(row['answer']).strip(),
        ))
    return problems


def sample_inspection(problems: list, n: int = 3) -> str:
    """Req 34.1/34.2: quick human-readable inspection dump."""
    lines = []
    for p in random.sample(problems, min(n, len(problems))):
        lines.append(f'--- {p.prompt_id} ---\nQ: {p.prompt_text[:300]}\nGT: {p.ground_truth}\n')
    return '\n'.join(lines)


print('✅ data module defined')# ════════════════════════════════════════════════════════════════════════════
# clustering.py — Prompt clustering for CB-GRPO
# Implements: Req 5 (semantic), Req 43 (difficulty-aware ablation)
# ════════════════════════════════════════════════════════════════════════════

def embed_prompts(prompts: list, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                  batch_size: int = 64) -> np.ndarray:
    """Req 5.1. Needs network access. Run once and cache."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(prompts, batch_size=batch_size,
                               show_progress_bar=True, convert_to_numpy=True,
                               normalize_embeddings=True)
    return embeddings.astype(np.float32)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int,
                        basis: str = 'semantic',
                        solve_rates: Optional[np.ndarray] = None,
                        difficulty_weight: float = 1.0,
                        random_state: int = 0) -> np.ndarray:
    """
    Req 5.2 (semantic) and Req 43.2 (difficulty-aware).

    basis='semantic'          → KMeans on embeddings only.
    basis='difficulty_aware'  → KMeans on [embeddings, difficulty_weight * solve_rate],
        concatenated, so prompts with different topics but similar difficulty
        can end up in the same cluster.
    """
    if basis == 'semantic':
        features = embeddings
    elif basis == 'difficulty_aware':
        if solve_rates is None:
            raise ValueError(
                'difficulty_aware clustering requires per-prompt solve_rates '
                '(Req 43.1 — reuse StaticSELFGate precomputation).'
            )
        sr = np.asarray(solve_rates, dtype=np.float64).reshape(-1, 1)
        sr = (sr - sr.mean()) / (sr.std() + 1e-8)  # standardize
        features = np.concatenate([embeddings, difficulty_weight * sr], axis=1)
    else:
        raise ValueError(f'Unknown clustering basis: {basis}')

    if len(features) < n_clusters:
        raise ValueError(f'n_clusters={n_clusters} exceeds number of prompts={len(features)}.')

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(features).astype(np.int64)


def cluster_size_distribution(cluster_ids: np.ndarray, n_clusters: int) -> dict:
    """Req 5.6."""
    counts = np.bincount(cluster_ids, minlength=n_clusters)
    return {int(c): int(n) for c, n in enumerate(counts)}


def save_cluster_assignments(cluster_ids: np.ndarray, prompt_ids: list, path: str) -> None:
    """Req 5.4."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({'prompt_ids': prompt_ids, 'cluster_ids': cluster_ids.tolist()}))


def load_cluster_assignments(path: str) -> tuple:
    """Req 5.5."""
    payload = json.loads(Path(path).read_text())
    return payload['prompt_ids'], np.asarray(payload['cluster_ids'], dtype=np.int64)


print('✅ clustering module defined')# ════════════════════════════════════════════════════════════════════════════
# evaluation.py — Pass@k, shrinkage slope, multi-seed significance testing
# Implements: Req 15, 16, 41 (ceiling check), 42 (multi-seed CI)
# ════════════════════════════════════════════════════════════════════════════

def pass_at_k_unbiased(num_samples: int, num_correct: int, k: int) -> float:
    """Req 15.4: Chen et al. (2021) unbiased estimator."""
    n, c = num_samples, num_correct
    if k > n:
        raise ValueError(f'k={k} cannot exceed n={n}.')
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k, exact=True) / comb(n, k, exact=True)


def compute_pass_at_k(correctness_matrix: np.ndarray, k_values: Sequence) -> dict:
    """correctness_matrix: shape [num_problems, n]. Returns {k: mean Pass@k}."""
    n = correctness_matrix.shape[1]
    num_correct = correctness_matrix.sum(axis=1).astype(int)
    return {k: float(np.mean([pass_at_k_unbiased(n, int(c), k) for c in num_correct]))
            for k in k_values}


def check_ceiling_effect(pass_at_k: dict, threshold: float = 0.90) -> dict:
    """Req 41.2: flag k-values where base Pass@k >= threshold (ceiling limited)."""
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
    """Req 16: linear regression of log(k) vs Δ Pass@k, with bootstrap 95% CI."""
    k_values = sorted(base_pass_k.keys())
    if len(k_values) < 2:
        raise ValueError('Need at least 2 k-values to fit a slope.')
    delta  = np.array([rl_pass_k[k] - base_pass_k[k] for k in k_values])
    log_k  = np.log(np.array(k_values, dtype=np.float64))
    slope, intercept = np.polyfit(log_k, delta, deg=1)
    predicted = slope * log_k + intercept
    ss_res = np.sum((delta - predicted) ** 2)
    ss_tot = np.sum((delta - delta.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    rng = np.random.default_rng(seed)
    idx_pool = np.arange(len(k_values))
    boot_slopes = []
    for _ in range(n_bootstrap):
        idx = rng.choice(idx_pool, size=len(idx_pool), replace=True)
        if np.ptp(log_k[idx]) == 0:
            continue
        s, _ = np.polyfit(log_k[idx], delta[idx], deg=1)
        boot_slopes.append(s)
    boot_slopes = np.asarray(boot_slopes)
    ci_lo, ci_hi = (np.percentile(boot_slopes, [2.5, 97.5])
                    if len(boot_slopes) > 0 else (float('nan'), float('nan')))

    return ShrinkageSlopeResult(
        slope=float(slope), intercept=float(intercept), r_squared=float(r_squared),
        ci_lower=float(ci_lo), ci_upper=float(ci_hi),
        delta_pass_k={k: float(d) for k, d in zip(k_values, delta)}
    )


@dataclass
class DeltaSlopeTestResult:
    mean_delta: float
    ci_lower: float
    ci_upper: float
    excludes_zero: bool
    n_seeds: int


def bootstrap_delta_slope_ci(condition_slopes: Sequence, baseline_slopes: Sequence,
                              n_bootstrap: int = 5000, seed: int = 0) -> DeltaSlopeTestResult:
    """
    Req 42.3-42.4: bootstrap CI on mean(condition) - mean(baseline) across seeds.
    Raises if fewer than 2 seeds on either side (Req 42.1).
    """
    cond = np.asarray(condition_slopes, dtype=np.float64)
    base = np.asarray(baseline_slopes, dtype=np.float64)
    n = min(len(cond), len(base))
    if n < 2:
        raise ValueError(
            'Need at least 2 seeds per condition for a meaningful CI. '
            'With 1 seed, any improvement claim is unsupported (Req 42.1).'
        )
    rng = np.random.default_rng(seed)
    diffs = np.array([
        cond[rng.choice(n, size=n, replace=True)].mean() -
        base[rng.choice(n, size=n, replace=True)].mean()
        for _ in range(n_bootstrap)
    ])
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return DeltaSlopeTestResult(
        mean_delta=float(cond.mean() - base.mean()),
        ci_lower=float(ci_lo), ci_upper=float(ci_hi),
        excludes_zero=bool(ci_lo > 0 or ci_hi < 0),
        n_seeds=n
    )


def gini_coefficient(values: np.ndarray) -> float:
    """Req 24.4 + Req 40.6: cluster-spend Gini for imbalance measurement."""
    x = np.asarray(values, dtype=np.float64)
    if np.any(x < 0):
        x = x - x.min()
    if x.sum() == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cum = np.cumsum(sorted_x)
    return float((2 * np.sum(np.arange(1, n + 1) * sorted_x) - (n + 1) * cum[-1]) / (n * cum[-1]))


print('✅ evaluation module defined')# ════════════════════════════════════════════════════════════════════════════
# diagnostics.py — Mechanism-level diagnostics (Req 40 / N5)
# All heavy lifting runs on saved checkpoints — zero extra training cost.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class InterferenceSnapshot:
    delta_plus: float   # mean( log π_new(y+|x) - log π_old(y+|x) )  [SELF paper Def. 4.1]
    delta_sq:   float   # mean( (log π_new(y|x) - log π_old(y|x))² ) [influence magnitude]


def compute_interference(logprob_before: np.ndarray, logprob_after: np.ndarray) -> InterferenceSnapshot:
    """
    Req 40.2-40.3: adapts SELF paper Definition 4.1 to a checkpoint pair.
    logprob_before/after: per-example mean per-token log-probs of base-model-correct
    completions under two policy snapshots. Negative delta_plus = negative interference.
    """
    before = np.asarray(logprob_before, dtype=np.float64)
    after  = np.asarray(logprob_after,  dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError('before/after arrays must have the same shape (same probing set).')
    delta = after - before
    return InterferenceSnapshot(delta_plus=float(delta.mean()), delta_sq=float(np.mean(delta ** 2)))


def entropy_from_probs(probs: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy (nats) of a categorical distribution given as probabilities."""
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def mean_token_entropy(token_entropies: Sequence) -> float:
    """
    Req 40.4: average entropy across variable-length sequences.
    Averages within-sequence first, then across batch (equal weight per sequence).
    """
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
    """
    Req 40.6-40.7: exploratory Pearson/Spearman correlation.
    With ~8-12 conditions, this is DESCRIPTIVE ONLY — p-values are unreliable
    and the summary report must state this limitation explicitly.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')
    if n < 3:
        pr = pp = sr = sp = float('nan')
    else:
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
    note = (f'n={n} conditions{(": " + label) if label else ""}. '
            f'{"Too few points for any meaningful correlation; report the scatter, not a coefficient." if n < 5 else "Exploratory/descriptive only — do not interpret as a hypothesis test given the small n."}')
    return CorrelationResult(pearson_r=float(pr), pearson_p=float(pp),
                              spearman_r=float(sr), spearman_p=float(sp), n=n, note=note)


# ── Torch-dependent helpers (lazy import so module is usable offline) ───────

def compute_probing_logprobs(model, tokenizer, probing_examples: list, device: str = 'cuda',
                              batch_size: int = 8) -> np.ndarray:
    """
    Req 40.1-40.2: evaluate mean per-token log-probability of base-model-correct
    completions (probing_examples: [{"prompt": str, "completion": str}]) under
    the CURRENT model weights. Call once per checkpoint snapshot.
    """
    import torch
    model.eval()
    logprobs = []
    with torch.no_grad():
        for i in range(0, len(probing_examples), batch_size):
            batch = probing_examples[i:i + batch_size]
            full_texts  = [ex['prompt'] + ex['completion'] for ex in batch]
            prompt_lens = [len(tokenizer(ex['prompt'], add_special_tokens=False).input_ids)
                           for ex in batch]
            enc = tokenizer(full_texts, return_tensors='pt',
                             padding=True, truncation=True).to(device)
            out = model(**enc)
            lp_full  = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
            target   = enc['input_ids'][:, 1:]
            tok_lp   = torch.gather(lp_full, 2, target.unsqueeze(-1)).squeeze(-1)
            mask     = enc['attention_mask'][:, 1:].float()
            for row in range(len(batch)):
                start  = max(prompt_lens[row] - 1, 0)
                rmask  = mask[row, start:]
                rlp    = tok_lp[row, start:]
                denom  = rmask.sum().clamp(min=1.0)
                logprobs.append(((rlp * rmask).sum() / denom).item())
    model.train()
    return np.asarray(logprobs, dtype=np.float64)


def token_entropy_from_logits_torch(logits) -> np.ndarray:
    """Req 40.4: per-position entropy from logits [batch, seq, vocab]."""
    import torch
    probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    return entropy_from_probs(probs, axis=-1)


print('✅ diagnostics module defined')# ════════════════════════════════════════════════════════════════════════════
# gates.py — Gate strategies for GRPO advantage gating
# Implements:
#   VanillaGate      (Req 6)
#   CBGRPOGate       (Req 7, refined: spend = max(advantage,0) per addendum §4.1)
#   OSELFGate        (Req 8, online per-prompt EMA solve-rate)
#   CompositeGate    (Req 39 / N4: H-CB-GRPO = macro × micro)
#
# All gates: soft gating only (weights in [0,1]), no hard exclusion.
# All gates: distributed-sync-safe via local_increments + apply_synced_increments.
# ════════════════════════════════════════════════════════════════════════════

def _to_numpy(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _to_int_numpy(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.int64)


SyncFn = Callable[[np.ndarray], np.ndarray]


def identity_sync(x: np.ndarray) -> np.ndarray:
    return x  # no-op for single-process / unit tests


class Gate(ABC):
    """Strategy interface. All gates return per-sample weights in [0,1]."""
    name: str = 'abstract'

    @abstractmethod
    def compute_weights(self, cluster_ids: Sequence, prompt_ids: Sequence) -> np.ndarray: ...

    @abstractmethod
    def local_increments(self, cluster_ids, prompt_ids, advantages, is_correct) -> dict: ...

    @abstractmethod
    def apply_synced_increments(self, synced: dict) -> None: ...

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, sd: dict) -> None: ...

    def step(self, cluster_ids, prompt_ids, advantages, is_correct,
             sync_fn: SyncFn = identity_sync) -> np.ndarray:
        """Compute weights from current state, then update state with synced increments."""
        cluster_ids = _to_int_numpy(cluster_ids)
        prompt_ids  = _to_int_numpy(prompt_ids)
        weights = self.compute_weights(cluster_ids, prompt_ids)
        incr    = self.local_increments(cluster_ids, prompt_ids, advantages, is_correct)
        synced  = {k: sync_fn(v) for k, v in incr.items()}
        self.apply_synced_increments(synced)
        return weights


# ── VanillaGate ─────────────────────────────────────────────────────────────

class VanillaGate(Gate):
    name = 'vanilla'
    def compute_weights(self, cluster_ids, prompt_ids) -> np.ndarray:
        return np.ones(len(cluster_ids), dtype=np.float64)
    def local_increments(self, cluster_ids, prompt_ids, advantages, is_correct) -> dict:
        return {}
    def apply_synced_increments(self, synced: dict) -> None:
        pass
    def state_dict(self) -> dict:
        return {}
    def load_state_dict(self, sd: dict) -> None:
        pass


# ── CBGRPOGate (macro, positive-mass refined) ────────────────────────────────

class CBGRPOGate(Gate):
    """
    Req 7, refined per novelty addendum §4.1 / Req 39.2:
    spend = max(advantage, 0)  — only throttles winner-reinforcement,
    not corrective negative gradient on hard clusters.
    Set use_positive_mass_only=False to reproduce original |advantage| behavior.
    """
    name = 'cb_grpo'

    def __init__(self, n_clusters: int, theta: float = 1.5, decay: float = 0.98,
                 ema_alpha: float = 0.05, use_positive_mass_only: bool = True):
        self.n_clusters             = n_clusters
        self.theta                  = theta
        self.decay                  = decay
        self.ema_alpha              = ema_alpha
        self.use_positive_mass_only = use_positive_mass_only
        self.spend_ema              = np.zeros(n_clusters, dtype=np.float64)

    def compute_weights(self, cluster_ids, prompt_ids) -> np.ndarray:
        mean_spend = self.spend_ema.mean() + 1e-8
        ratio  = self.spend_ema[cluster_ids] / mean_spend
        over   = np.clip(ratio - self.theta, a_min=0.0, a_max=None)
        return np.where(ratio > self.theta, self.decay ** over, 1.0)

    def local_increments(self, cluster_ids, prompt_ids, advantages, is_correct) -> dict:
        adv  = _to_numpy(advantages)
        mass = np.maximum(adv, 0.0) if self.use_positive_mass_only else np.abs(adv)
        sum_  = np.zeros(self.n_clusters, dtype=np.float64)
        count = np.zeros(self.n_clusters, dtype=np.float64)
        np.add.at(sum_,  cluster_ids, mass)
        np.add.at(count, cluster_ids, 1.0)
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
        """So cb_grpo-alone runs also feed the Gini/shrinkage correlation in
        section 14, not only h_cb_grpo runs (CompositeGate already had this)."""
        return {'cluster_spend_ema': self.spend_ema.tolist(),
                'gini': gini_coefficient(self.spend_ema)}


# ── OSELFGate (micro) ────────────────────────────────────────────────────────

class OSELFGate(Gate):
    """
    Req 8: online EMA solve-rate per prompt. Down-weights prompts the model
    has already effectively solved (winner-take-all suppression at the
    per-prompt level, which is the mechanism identified by arXiv:2510.02230).
    """
    name = 'o_self'

    def __init__(self, n_prompts: int, tau_solve: float = 0.7, lambda_self: float = 0.3,
                 ema_alpha: float = 0.10):
        self.n_prompts   = n_prompts
        self.tau_solve   = tau_solve
        self.lambda_self = lambda_self
        self.ema_alpha   = ema_alpha
        self.solve_ema   = np.zeros(n_prompts, dtype=np.float64)

    def compute_weights(self, cluster_ids, prompt_ids) -> np.ndarray:
        solved = self.solve_ema[prompt_ids] > self.tau_solve
        return np.where(solved, self.lambda_self, 1.0)

    def local_increments(self, cluster_ids, prompt_ids, advantages, is_correct) -> dict:
        correct  = _to_numpy(is_correct)
        sum_c    = np.zeros(self.n_prompts, dtype=np.float64)
        count    = np.zeros(self.n_prompts, dtype=np.float64)
        np.add.at(sum_c, prompt_ids, correct)
        np.add.at(count, prompt_ids, 1.0)
        return {'self_correct_sum': sum_c, 'self_count': count}

    def apply_synced_increments(self, synced: dict) -> None:
        s, c  = synced['self_correct_sum'], synced['self_count']
        touched = c > 0
        rate    = np.zeros(self.n_prompts, dtype=np.float64)
        rate[touched] = s[touched] / c[touched]
        self.solve_ema[touched] = (self.ema_alpha * rate[touched]
                                    + (1 - self.ema_alpha) * self.solve_ema[touched])

    def state_dict(self) -> dict:
        return {'solve_ema': self.solve_ema.copy()}

    def load_state_dict(self, sd: dict) -> None:
        self.solve_ema = np.asarray(sd['solve_ema'], dtype=np.float64).copy()


# ── CompositeGate (H-CB-GRPO, Req 39 / N4) ──────────────────────────────────

def _combine(macro_w: np.ndarray, micro_w: np.ndarray, op: str) -> np.ndarray:
    """Req 39.5: configurable combination operator for ablation."""
    if op == 'multiplicative':  return macro_w * micro_w
    if op == 'min':             return np.minimum(macro_w, micro_w)
    if op == 'harmonic_mean':   return 2 * macro_w * micro_w / (macro_w + micro_w + 1e-8)
    raise ValueError(f'Unknown combine op: {op}')


class CompositeGate(Gate):
    """
    Req 39 / N4 — H-CB-GRPO.
    Wraps CBGRPOGate (macro) + OSELFGate (micro). No new state machinery —
    only combines what the two children already track.
    Implements the algorithm from novelty_upgrade_addendum.md §4.1.
    """
    name = 'h_cb_grpo'

    def __init__(self, macro: CBGRPOGate, micro: OSELFGate, combine_op: str = 'multiplicative'):
        self.macro      = macro
        self.micro      = micro
        self.combine_op = combine_op

    def compute_weights(self, cluster_ids, prompt_ids) -> np.ndarray:
        macro_w = self.macro.compute_weights(cluster_ids, prompt_ids)
        micro_w = self.micro.compute_weights(cluster_ids, prompt_ids)
        return _combine(macro_w, micro_w, self.combine_op)

    def local_increments(self, cluster_ids, prompt_ids, advantages, is_correct) -> dict:
        macro_i = self.macro.local_increments(cluster_ids, prompt_ids, advantages, is_correct)
        micro_i = self.micro.local_increments(cluster_ids, prompt_ids, advantages, is_correct)
        merged  = {f'macro__{k}': v for k, v in macro_i.items()}
        merged.update({f'micro__{k}': v for k, v in micro_i.items()})
        return merged

    def apply_synced_increments(self, synced: dict) -> None:
        macro_p = {k[len('macro__'):]: v for k, v in synced.items() if k.startswith('macro__')}
        micro_p = {k[len('micro__'):]: v for k, v in synced.items() if k.startswith('micro__')}
        self.macro.apply_synced_increments(macro_p)
        self.micro.apply_synced_increments(micro_p)

    def state_dict(self) -> dict:
        return {'macro': self.macro.state_dict(), 'micro': self.micro.state_dict(),
                'combine_op': self.combine_op}

    def load_state_dict(self, sd: dict) -> None:
        self.macro.load_state_dict(sd['macro'])
        self.micro.load_state_dict(sd['micro'])
        self.combine_op = sd.get('combine_op', self.combine_op)

    def diagnostic_snapshot(self) -> dict:
        """Req 39.6: per-step diagnostic logging for macro/micro weights."""
        return {
            'cluster_spend_ema': self.macro.spend_ema.tolist(),
            'gini': gini_coefficient(self.macro.spend_ema),
            'prompt_solve_ema_mean': float(self.micro.solve_ema.mean()),
            'frac_solve_above_tau': float((self.micro.solve_ema > self.micro.tau_solve).mean()),
        }


def build_gate(gate_type: str, n_clusters: int, n_prompts: int, cfg: GateConfig) -> Gate:
    """Factory used by training loop."""
    if gate_type == 'vanilla':   return VanillaGate()
    if gate_type == 'cb_grpo':   return CBGRPOGate(n_clusters, theta=cfg.cb_theta,
                                                    decay=cfg.cb_decay, ema_alpha=cfg.cb_ema_alpha,
                                                    use_positive_mass_only=cfg.cb_use_positive_mass_only)
    if gate_type == 'o_self':    return OSELFGate(n_prompts, tau_solve=cfg.self_tau_solve,
                                                   lambda_self=cfg.self_lambda, ema_alpha=cfg.self_ema_alpha)
    if gate_type == 'h_cb_grpo':
        macro = CBGRPOGate(n_clusters, theta=cfg.cb_theta, decay=cfg.cb_decay,
                            ema_alpha=cfg.cb_ema_alpha,
                            use_positive_mass_only=cfg.cb_use_positive_mass_only)
        micro = OSELFGate(n_prompts, tau_solve=cfg.self_tau_solve,
                           lambda_self=cfg.self_lambda, ema_alpha=cfg.self_ema_alpha)
        return CompositeGate(macro, micro, combine_op=cfg.combine_op)
    raise ValueError(f'Unknown gate_type: {gate_type}')


print('✅ gates module defined (VanillaGate, CBGRPOGate, OSELFGate, CompositeGate)')# ════════════════════════════════════════════════════════════════════════════
# model_utils.py — QLoRA model loading, rollout generation, log-prob scoring
# Implements: Req 3 (QLoRA), Req 6.2 (rollouts), Req 16 (reference log-probs)
# GPU required. Skip this cell if running self-tests only.
# ════════════════════════════════════════════════════════════════════════════

QWEN_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj']


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qlora_model(model_name: str, local_process_index: int,
                      lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
                      gradient_checkpointing: bool = True):
    """
    Req 3.1-3.3: 4-bit NF4 + QLoRA.
    CRITICAL: use device_map={'': local_process_index}, NOT 'auto', for
    DDP on Kaggle 2×T4 (prevents model sharding vs process sharding conflict).
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map={'': local_process_index}, torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           target_modules=QWEN_TARGET_MODULES, bias='none', task_type='CAUSAL_LM')
    return get_peft_model(model, lora_cfg)


def report_vram(tag: str = '') -> str:
    """Req 3.4, 28.4: VRAM usage per GPU."""
    import torch
    if not torch.cuda.is_available():
        return f'[{tag}] CUDA not available.'
    dev = torch.cuda.current_device()
    used = torch.cuda.memory_allocated(dev) / 1e9
    rsv  = torch.cuda.memory_reserved(dev)  / 1e9
    return f'[{tag}] GPU{dev}: allocated={used:.2f}GB  reserved={rsv:.2f}GB'


@dataclass
class RolloutBatch:
    prompt_ids:   list
    cluster_ids:  list
    ground_truths: list
    chat_prompts: list
    completions:  list
    group_size:   int


def generate_rollouts(model, tokenizer, prompt_records: list, group_size: int,
                       max_new_tokens: int, temperature: float, top_p: float,
                       microbatch_prompts: int = 4) -> RolloutBatch:
    """
    Req 6.2 / 31.2: generate `group_size` rollouts per prompt.
    Call with unwrapped model (accelerator.unwrap_model) — DDP hooks
    are incompatible with .generate().
    """
    import torch
    was_training = model.training
    model.eval()
    prev_padding = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    out_pid, out_cid, out_gt, out_cp, out_comps = [], [], [], [], []

    with torch.no_grad():
        for start in range(0, len(prompt_records), microbatch_prompts):
            chunk = prompt_records[start:start + microbatch_prompts]
            expanded = [r['chat_prompt'] for r in chunk for _ in range(group_size)]
            enc = tokenizer(expanded, return_tensors='pt',
                             padding=True, truncation=True).to(model.device)
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=temperature, top_p=top_p,
                pad_token_id=tokenizer.pad_token_id
            )
            input_len  = enc['input_ids'].shape[1]
            completions = tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True)
            for i, r in enumerate(chunk):
                for g in range(group_size):
                    out_pid.append(r['prompt_id'])
                    out_cid.append(r['cluster_id'])
                    out_gt.append(r['ground_truth'])
                    out_cp.append(r['chat_prompt'])
                    out_comps.append(completions[i * group_size + g])

    tokenizer.padding_side = prev_padding
    if was_training:
        model.train()
    return RolloutBatch(prompt_ids=out_pid, cluster_ids=out_cid, ground_truths=out_gt,
                         chat_prompts=out_cp, completions=out_comps, group_size=group_size)


def score_sequences(model, tokenizer, chat_prompts, completions, max_length=1024):
    import torch
    tokenizer.padding_side = 'right'
    full_texts = [p + c for p, c in zip(chat_prompts, completions)]
    prompt_lens = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in chat_prompts]

    enc = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True,
                     max_length=max_length).to(model.device)
    out = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])

    logits = out.logits[:, :-1, :]            # keep model dtype (bf16) -- do NOT .float() the whole tensor
    target_ids = enc['input_ids'][:, 1:]

    # FIX: cross_entropy's fused kernel computes log_softmax + NLL internally
    # without ever materializing a separate [batch, seq_len, vocab] float32
    # tensor the way log_softmax(...).gather(...) does. Same math, far less
    # peak memory -- this is what was OOMing on a 152k-token vocab.
    token_nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction='none',
    ).view(target_ids.shape)
    token_logprobs = -token_nll                # [B, L-1]

    attn = enc['attention_mask'][:, 1:]
    seq_len = attn.shape[1]
    completion_mask = torch.zeros_like(attn)
    for row, plen in enumerate(prompt_lens):
        start = max(min(plen - 1, seq_len), 0)
        completion_mask[row, start:] = attn[row, start:]

    del out, logits    # release the big logits tensor before returning, not at next GC cycle
    return token_logprobs, completion_mask.float()

def reference_logprobs(model, tokenizer, chat_prompts: list, completions: list, max_length: int = 1024):
    """KL/reference log-probs using LoRA disable_adapter (no second model copy needed)."""
    import torch
    with torch.no_grad(), model.disable_adapter():
        return score_sequences(model, tokenizer, chat_prompts, completions, max_length)


print('✅ model_utils module defined')# ════════════════════════════════════════════════════════════════════════════
# grpo_core.py — GRPO advantage computation, loss, and full training step
# From-scratch implementation (not TRL wrapper) so gating works across TRL
# version changes. Implements: Req 6 (GRPO), Req 7 (gated loss), Req 39.
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class StepMetrics:
    loss: float
    mean_reward: float
    format_success_rate: float
    correctness_success_rate: float
    mean_gate_weight: float
    frac_gated_below_1: float
    approx_kl: float
    extra: dict = field(default_factory=dict)


def compute_group_advantages(rewards, group_size: int):
    """
    GRPO group-relative advantage (design.md Algorithm / SELF paper Eq. 10).
    rewards: flat tensor [num_prompts * group_size].
    Degenerate zero-variance group → all-zero advantage (no learning signal).
    """
    import torch
    n_prompts = rewards.shape[0] // group_size
    r   = rewards.view(n_prompts, group_size)
    mu  = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, unbiased=True, keepdim=True) if group_size > 1 else torch.ones_like(mu)
    return ((r - mu) / (std + 1e-4)).view(-1)


def grpo_policy_loss(new_logprobs, old_logprobs, advantages, completion_mask, clip_eps: float):
    """PPO-clip objective, token-level, matching Eq. 11 of arXiv:2510.02230 appendix."""
    import torch
    ratio     = torch.exp(new_logprobs - old_logprobs)        # [B, L]
    adv       = advantages.unsqueeze(-1)                       # [B, 1]
    unclipped = ratio * adv
    clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    per_token = -torch.minimum(unclipped, clipped)
    denom     = completion_mask.sum().clamp(min=1.0)
    return (per_token * completion_mask).sum() / denom


def approx_kl_k3(new_logprobs, ref_logprobs, completion_mask):
    """k3 estimator: KL(π || ref) ≈ exp(log_ref - log_π) - (log_ref - log_π) - 1."""
    import torch
    diff  = ref_logprobs - new_logprobs
    kl    = torch.exp(diff) - diff - 1.0
    denom = completion_mask.sum().clamp(min=1.0)
    return (kl * completion_mask).sum() / denom


def run_training_step(model, tokenizer, accelerator, gate: Gate,
                       prompt_records: list, group_size: int,
                       max_new_tokens: int, temperature: float, top_p: float,
                       gen_microbatch_prompts: int, clip_eps: float,
                       kl_beta: float, num_ppo_epochs: int,
                       reward_mode: str, format_weight: float, correctness_weight: float,
                       sync_fn: Callable = identity_sync) -> StepMetrics:
    """
    One full GRPO(+gate) update for the local process's prompt shard.
    `sync_fn` (built from accelerator.reduce in train loop) keeps gate state
    in sync across GPUs — see gates.py design note for why both DDP and
    gate-sync are necessary.
    """
    import torch

    # 1. Rollouts (unwrapped — DDP hooks incompatible with .generate())
    unwrapped = accelerator.unwrap_model(model)
    rollouts  = generate_rollouts(unwrapped, tokenizer, prompt_records, group_size,
                                   max_new_tokens, temperature, top_p, gen_microbatch_prompts)

    # 2. Rewards (CPU, sympy)
    breakdowns = [
        compute_reward(comp, gt, reward_mode, format_weight, correctness_weight)
        for comp, gt in zip(rollouts.completions, rollouts.ground_truths)
    ]
    rewards_t  = torch.tensor([b.total for b in breakdowns], dtype=torch.float32, device=model.device)
    is_correct = np.array([1.0 if b.is_correct else 0.0 for b in breakdowns])

    # 3. Group-relative advantages
    advantages = compute_group_advantages(rewards_t, group_size)

    # 4. Gate: compute weights from current (pre-update) state, then sync-update
    gw_np      = gate.step(rollouts.cluster_ids, rollouts.prompt_ids,
                            advantages.detach().cpu().numpy(), is_correct, sync_fn=sync_fn)
    gate_w     = torch.tensor(gw_np, dtype=torch.float32, device=model.device)
    gated_adv  = advantages * gate_w

    # 5. Old-policy / reference log-probs
    with torch.no_grad():
        old_lp, cmask = score_sequences(unwrapped, tokenizer,
                                         rollouts.chat_prompts, rollouts.completions)
        ref_lp = None
        if kl_beta > 0:
            ref_lp, _ = reference_logprobs(unwrapped, tokenizer,
                                             rollouts.chat_prompts, rollouts.completions)

    # 6. Policy gradient (num_ppo_epochs ≥ 1 reuses this rollout batch)
    last_loss = None
    last_kl   = 0.0
    for _ in range(max(1, num_ppo_epochs)):
        new_lp, cmask = score_sequences(model, tokenizer,
                                         rollouts.chat_prompts, rollouts.completions)
        loss = grpo_policy_loss(new_lp, old_lp, gated_adv, cmask, clip_eps)
        if kl_beta > 0 and ref_lp is not None:
            kl   = approx_kl_k3(new_lp, ref_lp, cmask)
            loss = loss + kl_beta * kl
            last_kl = float(kl.detach().item())
        accelerator.backward(loss)
        last_loss = loss

    return StepMetrics(
        loss=float(last_loss.detach().item()),
        mean_reward=float(rewards_t.mean().item()),
        format_success_rate=float(np.mean([b.is_well_formed for b in breakdowns])),
        correctness_success_rate=float(np.mean([b.is_correct for b in breakdowns])),
        mean_gate_weight=float(gw_np.mean()),
        frac_gated_below_1=float((gw_np < 0.999).mean()),
        approx_kl=last_kl,
    )


print('✅ grpo_core module defined')# ════════════════════════════════════════════════════════════════════════════
# train_helpers — PromptSampler, checkpointing, Pass@k eval
# ════════════════════════════════════════════════════════════════════════════

class PromptSampler:
    """
    Communication-free distributed prompt sharding.
    Every process independently computes the identical epoch permutation
    (same seed → same numpy permutation) and reads its own disjoint slice
    via process_index. Resumable from global_step alone.
    """
    def __init__(self, n_prompts: int, per_device_bs: int, num_processes: int,
                 process_index: int, seed: int):
        self.n              = n_prompts
        self.per_device_bs  = per_device_bs
        self.num_processes  = num_processes
        self.process_index  = process_index
        self.seed           = seed
        self.global_bs      = per_device_bs * num_processes
        self.steps_per_epoch = max(n_prompts // self.global_bs, 1)
        self._cached_epoch  = None
        self._order         = None

    def _order_for_epoch(self, epoch: int) -> np.ndarray:
        if epoch != self._cached_epoch:
            self._order = np.random.default_rng(self.seed + epoch).permutation(self.n)
            self._cached_epoch = epoch
        return self._order

    def get_batch(self, global_step: int) -> list:
        epoch = global_step // self.steps_per_epoch
        within = global_step % self.steps_per_epoch
        order  = self._order_for_epoch(epoch)
        start  = within * self.global_bs + self.process_index * self.per_device_bs
        return order[start:start + self.per_device_bs].tolist()


def save_run_state(run_dir: Path, global_step: int, gate: Gate, extra: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {'global_step': global_step, 'gate_state': gate.state_dict(), **extra}
    def _default(o):
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f'not JSON serializable: {type(o)}')
    (run_dir / 'run_state.json').write_text(json.dumps(payload, default=_default, indent=2))


def load_run_state(run_dir: Path) -> Optional[dict]:
    path = run_dir / 'run_state.json'
    return json.loads(path.read_text()) if path.exists() else None


def run_pass_at_k_eval(model, tokenizer, eval_problems: list,
                        n_samples: int, k_values: list,
                        temperature: float, top_p: float,
                        max_new_tokens: int, microbatch_prompts: int) -> tuple:
    """
    Req 41.3: n up to eval_samples_per_problem (default 30, not 10) so
    shrinkage is measurable well beyond k=10.
    """
    records = [{'prompt_id': i, 'cluster_id': 0,
                 'chat_prompt': p.chat_prompt, 'ground_truth': p.ground_truth}
                for i, p in enumerate(eval_problems)]
    rollouts = generate_rollouts(model, tokenizer, records, group_size=n_samples,
                                  max_new_tokens=max_new_tokens, temperature=temperature,
                                  top_p=top_p, microbatch_prompts=microbatch_prompts)
    correctness = np.array([
        1 if answers_equivalent(parse_output(c).boxed, gt) else 0
        for c, gt in zip(rollouts.completions, rollouts.ground_truths)
    ]).reshape(len(eval_problems), n_samples)
    return compute_pass_at_k(correctness, k_values), correctness.shape


print('✅ train_helpers defined (PromptSampler, checkpointing, Pass@k eval)')
# ════════════════════════════════════════════════════════════════════════════
# STANDALONE SCRIPT ENTRY POINT
# This is what actually gets multi-process, multi-GPU behavior: `accelerate
# launch` spawns N independent OS processes that each import and run this
# file from the top and then call main(). A plain Jupyter cell runs in ONE
# already-existing process, so Accelerator() inside a notebook cell is
# always a world_size=1 group regardless of accelerate_config.yaml -- that
# YAML only takes effect for processes spawned BY `accelerate launch` itself.
# This is why the module-definition cells above are necessary but not
# sufficient for real 2xT4 training: they define the classes/functions
# correctly, but running them inline in the notebook kernel cannot split
# work across 2 GPUs, no matter what the config file says.
# ════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", choices=["0.5B", "1.5B"], default="1.5B")
    p.add_argument("--gate_type", choices=["vanilla", "cb_grpo", "o_self", "h_cb_grpo"], default="h_cb_grpo")
    p.add_argument("--compute_tier", choices=["smoke", "standard", "final"], default="smoke")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clustering_basis", choices=["semantic", "difficulty_aware"], default="semantic")
    p.add_argument("--combine_op", choices=["multiplicative", "min", "harmonic_mean"], default="multiplicative")
    p.add_argument("--kl_beta", type=float, default=0.0)
    p.add_argument("--num_ppo_epochs", type=int, default=1)
    p.add_argument("--artifact_root", type=str, default="/kaggle/working/artifacts")
    args = p.parse_args()

    import torch
    from accelerate import Accelerator
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

    exp_name = f"exp_{args.gate_type}"   # matches the notebook's §12/§14 naming convention
    cfg = ExperimentConfig(
        exp_name=exp_name, model_size=args.model_size, compute_tier=args.compute_tier,
        seed=args.seed,
        gate=GateConfig(gate_type=args.gate_type, combine_op=args.combine_op,
                         cb_use_positive_mass_only=True),
        optim=OptimConfig(kl_beta=args.kl_beta, num_ppo_epochs=args.num_ppo_epochs),
        cluster=ClusterConfig(basis=args.clustering_basis,
                               cache_dir=f"{args.artifact_root}/clusters"),
        output_root=f"{args.artifact_root}/runs",
    )
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    from datetime import timedelta
    from accelerate.utils import InitProcessGroupKwargs
    # 2-hour process group timeout: baseline eval + periodic evals on rank-0
    # can take >10 min; default 600s NCCL watchdog kills rank-1 at the barrier.
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))]
    )

    def log(msg: str):
        if accelerator.is_main_process:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    seed = cfg.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    log(f"gate={cfg.gate.gate_type}  model={cfg.model_size}  tier={cfg.compute_tier}"
        f"  world_size={accelerator.num_processes}  (expect 2 on Kaggle 2xT4)")

    # main_process_first: rank-0 downloads/caches model, rank-1 waits,
    # then both read from the already-populated cache — prevents HF lock deadlock
    with accelerator.main_process_first():
        tokenizer = load_tokenizer(cfg.model_name)
        model = load_qlora_model(cfg.model_name, accelerator.local_process_index,
                                  cfg.optim.lora_r, cfg.optim.lora_alpha, cfg.optim.lora_dropout)
    log(report_vram("after model load"))

    train_problems = load_gsm8k(tokenizer, split="train")
    eval_problems = load_gsm8k(tokenizer, split="test")
    log(f"Loaded {len(train_problems)} train / {len(eval_problems)} eval problems.")

    cluster_cache = Path(cfg.cluster.cache_dir) / f"{cfg.model_size}_{cfg.cluster.basis}_clusters.json"
    if accelerator.is_main_process and not cluster_cache.exists():
        log("Computing prompt embeddings + clusters (main process only)...")
        embeddings = embed_prompts([p.prompt_text for p in train_problems], cfg.cluster.embedding_model)
        if cfg.cluster.basis == "difficulty_aware":
            sr_path = Path(args.artifact_root) / "solve_rates.json"
            if not sr_path.exists():
                raise RuntimeError(
                    "difficulty_aware clustering needs per-prompt solve rates. "
                    "Run notebook section 13 (clustering ablation) first to generate solve_rates.json."
                )
            solve_rates = np.array(json.loads(sr_path.read_text())["solve_rates"])
        else:
            solve_rates = None
        labels = cluster_embeddings(embeddings, cfg.cluster.n_clusters, basis=cfg.cluster.basis,
                                     solve_rates=solve_rates, random_state=cfg.cluster.random_state)
        save_cluster_assignments(labels, [p.prompt_id for p in train_problems], str(cluster_cache))
        log(f"Cluster sizes: {cluster_size_distribution(labels, cfg.cluster.n_clusters)}")
    accelerator.wait_for_everyone()
    _, cluster_labels = load_cluster_assignments(str(cluster_cache))

    prompt_records_all = [
        {"prompt_id": i, "cluster_id": int(cluster_labels[i]),
         "chat_prompt": p.chat_prompt, "ground_truth": p.ground_truth}
        for i, p in enumerate(train_problems)
    ]

    gate = build_gate(cfg.gate.gate_type, cfg.cluster.n_clusters, len(train_problems), cfg.gate)
    log(f"Gate: {gate.name}")

    def sync_fn(x: np.ndarray) -> np.ndarray:
        t = torch.tensor(x, dtype=torch.float64, device=accelerator.device)
        return accelerator.reduce(t, reduction="sum").cpu().numpy()

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.optim.learning_rate)
    total_steps = cfg.training_steps
    warmup_steps = int(cfg.optim.warmup_ratio * total_steps)
    scheduler = (get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
                 if cfg.optim.lr_scheduler_type == "cosine"
                 else get_constant_schedule_with_warmup(optimizer, warmup_steps))
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    sampler = PromptSampler(len(train_problems), cfg.optim.per_device_prompts_per_step,
                             accelerator.num_processes, accelerator.process_index, seed)

    start_step = 0
    prior = load_run_state(run_dir)
    ckpt_dir = run_dir / "accelerate_ckpt"
    if prior is not None and ckpt_dir.exists():
        log(f"Resuming from step {prior['global_step']}")
        accelerator.load_state(str(ckpt_dir))
        gate.load_state_dict(prior["gate_state"])
        start_step = prior["global_step"]

    # FIX applied: only strip seed (run_dir.parent), not seed+model_size
    # (run_dir.parent.parent), so 0.5B and 1.5B don't collide on one baseline file.
    baseline_path = run_dir.parent / "baseline_pass_at_k.json"
    if accelerator.is_main_process and start_step == 0 and not baseline_path.exists():
        log("Computing base-model Pass@k baseline...")
        unwrapped = accelerator.unwrap_model(model)
        # 50 problems x 10 samples ~ 2 min; keeps rank-0 well under NCCL timeout
        with unwrapped.disable_adapter():
            base_pk, _ = run_pass_at_k_eval(
                unwrapped, tokenizer, eval_problems[:50],
                n_samples=10, k_values=[1, 2, 3, 5, 10],
                temperature=cfg.generation.eval_temperature, top_p=cfg.generation.eval_top_p,
                max_new_tokens=cfg.generation.max_new_tokens,
                microbatch_prompts=1,
            )
        ceiling = check_ceiling_effect(base_pk)
        baseline_path.write_text(json.dumps({"pass_at_k": base_pk, "ceiling_flags": ceiling}, indent=2))
        log(f"Base Pass@k: {base_pk}")
        log(f"Ceiling flags: {ceiling}")
    accelerator.wait_for_everyone()

    log_path = run_dir / "train_log.jsonl"
    _train_start_time = time.time()
    log(f"Training: {cfg.training_steps} steps total, checkpointing every {cfg.checkpoint_interval} steps")
    for global_step in range(start_step, cfg.training_steps):
        step_metrics = []
        for micro in range(cfg.optim.grad_accum_steps):
            idx = sampler.get_batch(global_step * cfg.optim.grad_accum_steps + micro)
            batch_records = [prompt_records_all[i] for i in idx]
            with accelerator.accumulate(model):
                metrics = run_training_step(
                    model, tokenizer, accelerator, gate, batch_records,
                    group_size=cfg.generation.rollouts_per_prompt,
                    max_new_tokens=cfg.generation.max_new_tokens,
                    temperature=cfg.generation.temperature, top_p=cfg.generation.top_p,
                    gen_microbatch_prompts=cfg.generation.gen_microbatch_prompts,
                    clip_eps=cfg.optim.clip_eps, kl_beta=cfg.optim.kl_beta,
                    num_ppo_epochs=cfg.optim.num_ppo_epochs,
                    reward_mode=cfg.reward.reward_mode, format_weight=cfg.reward.format_weight,
                    correctness_weight=cfg.reward.correctness_weight, sync_fn=sync_fn,
                )
                step_metrics.append(metrics)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if global_step % cfg.log_interval == 0 and accelerator.is_main_process:
            agg = {
                "step": global_step,
                "loss": float(np.mean([m.loss for m in step_metrics])),
                "mean_reward": float(np.mean([m.mean_reward for m in step_metrics])),
                "format_success_rate": float(np.mean([m.format_success_rate for m in step_metrics])),
                "correctness_success_rate": float(np.mean([m.correctness_success_rate for m in step_metrics])),
                "mean_gate_weight": float(np.mean([m.mean_gate_weight for m in step_metrics])),
                "frac_gated_below_1": float(np.mean([m.frac_gated_below_1 for m in step_metrics])),
                "approx_kl": float(np.mean([m.approx_kl for m in step_metrics])),
                "lr": scheduler.get_last_lr()[0],
            }
            if hasattr(gate, "diagnostic_snapshot"):
                agg["gate_diagnostic"] = gate.diagnostic_snapshot()
            # Pretty human-readable progress line
            elapsed = time.time() - _train_start_time
            steps_done = global_step - start_step + 1
            steps_total = cfg.training_steps - start_step
            eta_sec = (elapsed / steps_done) * (steps_total - steps_done) if steps_done > 0 else 0
            eta_str = f"{int(eta_sec//3600):02d}h{int((eta_sec%3600)//60):02d}m{int(eta_sec%60):02d}s"
            pct = 100.0 * steps_done / steps_total
            log(
                f"step {global_step:4d}/{cfg.training_steps} ({pct:5.1f}%) | "
                f"loss={agg['loss']:.4f} | reward={agg['mean_reward']:+.3f} | "
                f"fmt={agg['format_success_rate']:.2f} | cor={agg['correctness_success_rate']:.2f} | "
                f"gate_w={agg['mean_gate_weight']:.3f} | kl={agg['approx_kl']:.4f} | "
                f"lr={agg['lr']:.2e} | ETA {eta_str}"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(agg) + "\n")

        if global_step > 0 and global_step % cfg.checkpoint_interval == 0:
            accelerator.wait_for_everyone()
            accelerator.save_state(str(ckpt_dir))
            if accelerator.is_main_process:
                save_run_state(run_dir, global_step, gate, {})
                # Belt-and-suspenders: accelerator.save_state()'s checkpoint format
                # for PeftModels isn't guaranteed identical to peft's own
                # save_pretrained() across every accelerate version. Also save a
                # peft-native adapter copy so §11 (interference) and eval scripts
                # can load it with PeftModel.from_pretrained(...) reliably.
                accelerator.unwrap_model(model).save_pretrained(str(run_dir / "adapter"))
            log(
                f"*** CHECKPOINT SAVED at step {global_step} "
                f"({100*global_step//cfg.training_steps}% done) -> {run_dir} ***"
            )

        if global_step > 0 and global_step % cfg.eval_interval == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            # Periodic eval: 50 problems keeps under NCCL timeout
            rl_pk, _ = run_pass_at_k_eval(
                unwrapped, tokenizer, eval_problems[:50],
                n_samples=10, k_values=[1, 2, 3, 5, 10],
                temperature=cfg.generation.eval_temperature, top_p=cfg.generation.eval_top_p,
                max_new_tokens=cfg.generation.max_new_tokens,
                microbatch_prompts=1,
            )
            baseline_data = json.loads(baseline_path.read_text())
            slope_res = compute_shrinkage_slope(baseline_data["pass_at_k"], rl_pk)
            log(f"[eval@{global_step}] Pass@k={rl_pk}  slope={slope_res.slope:.4f}"
                f"  CI[{slope_res.ci_lower:.4f},{slope_res.ci_upper:.4f}]")
            with open(run_dir / "eval_log.jsonl", "a") as f:
                f.write(json.dumps({"step": global_step, "pass_at_k": rl_pk,
                                     "shrinkage_slope": slope_res.slope,
                                     "ci_lower": slope_res.ci_lower, "ci_upper": slope_res.ci_upper}) + "\n")

    accelerator.wait_for_everyone()
    accelerator.save_state(str(ckpt_dir))
    if accelerator.is_main_process:
        save_run_state(run_dir, cfg.training_steps, gate, {})
        accelerator.unwrap_model(model).save_pretrained(str(run_dir / "adapter"))
        cfg.to_json(str(run_dir / "config.json"))
        log(f"Training complete. Artifacts in {run_dir}")


if __name__ == "__main__":
    main()

