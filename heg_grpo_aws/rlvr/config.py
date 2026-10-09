"""Experiment configuration (Req 2, 33, 39, 41-44). Single-GPU AWS edition.

Deviations from the original Kaggle spec are recorded in PREREGISTRATION.md and README.md
(MATH training data, real SELF selection, MEG as credit redistribution, correctness-only reward).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

GATE_TYPES = ('vanilla', 'cb_grpo', 'o_self', 'h_cb_grpo', 'meg', 'heg_grpo', 'bbg')
MODEL_SIZES = ('0.5B', '1.5B')
TRAIN_DATASETS = ('math', 'gsm8k')
PARTITION_METHODS = ('embedding', 'bigram')
CLUSTER_BASES = ('subject', 'semantic', 'difficulty_aware')

MODEL_NAME_BY_SIZE = {
    '0.5B': 'Qwen/Qwen2.5-0.5B-Instruct',
    '1.5B': 'Qwen/Qwen2.5-1.5B-Instruct',
}

TIER_SETTINGS: dict[str, dict[str, int]] = {
    # eval_interval=0 disables in-training monitor evals.
    'pilot':    {'training_steps': 30,   'checkpoint_interval': 15, 'eval_interval': 0,   'log_interval': 1},
    'standard': {'training_steps': 800,  'checkpoint_interval': 25, 'eval_interval': 100, 'log_interval': 5},
    'final':    {'training_steps': 1800, 'checkpoint_interval': 25, 'eval_interval': 200, 'log_interval': 10},
}

DEFAULT_ARTIFACT_ROOT = os.environ.get('ARTIFACT_ROOT', str(Path.home() / 'artifacts'))
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


@dataclass
class ClusterConfig:
    # 'subject': MATH's 7 subjects are the macro topics. 'semantic' / 'difficulty_aware' are
    # KMeans ablations (Req 5 / Req 43) and the only options for GSM8K.
    basis: str = 'subject'
    n_clusters: int = 16
    embedding_model: str = EMBED_MODEL
    random_state: int = 0


@dataclass
class GateConfig:
    gate_type: str = 'vanilla'
    # CB-GRPO (macro). The spec's theta=1.5, decay=0.98 never fired in the pilot (7 subjects: spend
    # ratios 0.38-1.46 after 30 steps, and the weight could not go below 0.98**5.5 = 0.90 anyway).
    # Pre-registered 2026-10-06: theta=1.1, decay chosen so a topic at 1.5x the mean spend gets half
    # the credit of an in-budget topic (decay**0.4 = 0.5 -> 0.18), mean-preserving over positives.
    cb_theta: float = 1.1
    cb_decay: float = 0.18
    cb_ema_alpha: float = 0.05
    cb_use_positive_mass_only: bool = True
    cb_mean_preserving: bool = True
    cb_clip_min: float = 0.3
    cb_clip_max: float = 3.0
    # SELF (2510.02230): weight for prompts whose greedy answer is already correct. 0 = hard filter.
    self_lambda: float = 0.0
    # MEG as rarity credit redistribution over correct rollouts (Cue-GRPO 2608.03467 rule)
    meg_alpha: float = 0.8
    meg_clip_min: float = 0.3
    meg_clip_max: float = 3.0
    # The real solution-mode partitioner (chosen by scripts/calibrate_meg.py); it also drives the
    # mode-entropy diagnostic logged for every condition.
    meg_partition: str = 'embedding'
    # Control condition: MEG's weights computed from random labels with the same per-group cluster count.
    meg_random_partition: bool = False
    meg_tau_mode: float = 0.85          # embedding: cosine-similarity threshold for "same mode"
    meg_bigram_threshold: float = 0.5   # bigram: Jaccard-distance threshold for "same mode"
    meg_embedding_model: str = EMBED_MODEL
    # BBG-style problem gate (2606.15455), GRPO-compatible variant
    bbg_k_ref: int = 16
    bbg_tau: float = 0.01
    combine_op: str = 'multiplicative'

    def validate(self, model_size: str) -> None:
        if self.gate_type not in GATE_TYPES:
            raise ValueError(f'Unknown gate_type {self.gate_type!r}; expected one of {GATE_TYPES}.')
        if not (0.0 <= self.self_lambda <= 1.0):
            raise ValueError('self_lambda must be in [0, 1].')
        if not (0.0 <= self.meg_alpha <= 1.0):
            raise ValueError('meg_alpha must be in [0, 1].')
        if not (0.0 < self.meg_clip_min <= 1.0 <= self.meg_clip_max):
            raise ValueError('need 0 < meg_clip_min <= 1 <= meg_clip_max.')
        if self.meg_partition not in PARTITION_METHODS:
            raise ValueError(f'meg_partition must be one of {PARTITION_METHODS}.')
        if not (0.0 < self.meg_tau_mode < 1.0):
            raise ValueError('meg_tau_mode must be in (0, 1).')
        if not (0.0 < self.meg_bigram_threshold < 1.0):
            raise ValueError('meg_bigram_threshold must be in (0, 1).')
        if self.cb_theta <= 1.0:
            raise ValueError('cb_theta must be > 1.0.')
        if not (0.0 < self.cb_decay < 1.0) or not (0.0 < self.cb_clip_min <= 1.0 <= self.cb_clip_max):
            raise ValueError('need 0 < cb_decay < 1 and 0 < cb_clip_min <= 1 <= cb_clip_max.')
        if self.bbg_k_ref < 1 or not (0.0 < self.bbg_tau < 1.0):
            raise ValueError('need bbg_k_ref >= 1 and 0 < bbg_tau < 1.')
        if self.combine_op not in ('multiplicative', 'min', 'harmonic_mean'):
            raise ValueError(f'Unknown combine_op {self.combine_op!r}.')


@dataclass
class RewardConfig:
    reward_mode: str = 'positive_only'
    # Binary correctness only. The strict tag-format reward was identically 0 for every rollout
    # (inert under group normalization); the format rate is still logged.
    format_weight: float = 0.0
    correctness_weight: float = 1.0


@dataclass
class GenerationConfig:
    rollouts_per_prompt: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    # Qwen's generation_config silently adds top_k=20 / repetition_penalty=1.05;
    # set explicitly so the sampling distribution is fully defined by this config.
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 1024
    # Sample from a merged copy of the LoRA model (rebuilt every step; 1.28x faster decoding on the A10G).
    merged_generation: bool = True
    monitor_eval_problems: int = 50
    monitor_eval_samples: int = 10
    monitor_eval_microbatch_prompts: int = 8


@dataclass
class OptimConfig:
    learning_rate: float = 1e-6
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    kl_beta: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 32
    # 0.0 keeps the update exactly on-policy (old == new log-probs), which also
    # lets the step skip a separate old-policy forward pass.
    lora_dropout: float = 0.0
    # All 16 prompts of an update are generated in one call (144 rows incl. greedy rows): measured
    # 78.9 ms per decode step vs 2 x 64.6 ms for two calls of 8 prompts (A10G, 2026-10-06).
    prompts_per_micro_step: int = 16
    grad_accum_steps: int = 1
    gradient_checkpointing: bool = False
    # Sequences per forward/backward chunk. 4 ran out of memory on 22 GB once completions reached
    # 1,024 tokens (pilot, 2026-10-06); 2 halves the activation and fp32-logit memory.
    score_chunk_size: int = 2


@dataclass
class EvalConfig:
    n_samples: int = 32
    k_values: list = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    temperature: float = 0.6
    top_p: float = 0.95
    microbatch_prompts: int = 8   # 256 rows per call: 1.19x faster than 4 on the A10G; 16 is slower
    sample_seed: int = 1234
    subset_seed: int = 0
    n_problems: dict = field(default_factory=lambda: {'gsm8k': 500, 'gsm8k_platinum': 500, 'math500': 500})
    max_new_tokens: dict = field(default_factory=lambda: {'gsm8k': 512, 'gsm8k_platinum': 512, 'math500': 1024})

    def settings_for(self, benchmark: str) -> dict:
        """The exact settings a base/run comparison must share."""
        return {
            'benchmark': benchmark, 'n_samples': self.n_samples, 'k_values': list(self.k_values),
            'temperature': self.temperature, 'top_p': self.top_p,
            'sample_seed': self.sample_seed, 'subset_seed': self.subset_seed,
            'n_problems': self.n_problems[benchmark], 'max_new_tokens': self.max_new_tokens[benchmark],
        }


@dataclass
class ExperimentConfig:
    model_size: str = '0.5B'
    compute_tier: str = 'standard'
    seed: int = 0
    tag: str = ''
    steps_override: Optional[int] = None
    train_dataset: str = 'math'
    math_levels: list = field(default_factory=lambda: [1, 2, 3, 4, 5])

    gate: GateConfig = field(default_factory=GateConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    artifact_root: str = DEFAULT_ARTIFACT_ROOT

    def __post_init__(self):
        if self.model_size not in MODEL_SIZES:
            raise ValueError(f'Unknown model_size {self.model_size!r}.')
        if self.compute_tier not in TIER_SETTINGS:
            raise ValueError(f'Unknown compute_tier {self.compute_tier!r}.')
        if self.train_dataset not in TRAIN_DATASETS:
            raise ValueError(f'train_dataset must be one of {TRAIN_DATASETS}.')
        if self.cluster.basis not in CLUSTER_BASES:
            raise ValueError(f'cluster basis must be one of {CLUSTER_BASES}.')
        if self.cluster.basis == 'subject' and self.train_dataset != 'math':
            raise ValueError("cluster basis 'subject' needs train_dataset='math' (GSM8K has no subjects).")
        self.gate.validate(self.model_size)
        if self.generation.rollouts_per_prompt < 2:
            raise ValueError('rollouts_per_prompt must be >= 2 (GRPO needs a group).')

    @property
    def model_name(self) -> str:
        return MODEL_NAME_BY_SIZE[self.model_size]

    @property
    def in_domain_benchmark(self) -> str:
        return 'math500' if self.train_dataset == 'math' else 'gsm8k'

    @property
    def training_steps(self) -> int:
        return self.steps_override or TIER_SETTINGS[self.compute_tier]['training_steps']

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
    def run_name(self) -> str:
        return f'{self.gate.gate_type}__{self.tag}' if self.tag else self.gate.gate_type

    @property
    def run_dir(self) -> Path:
        return run_dir_for(self.artifact_root, self.run_name, self.model_size, self.seed)

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
        raw['eval'] = EvalConfig(**raw.pop('eval'))
        return ExperimentConfig(**raw)


def run_dir_for(artifact_root: str, run_name: str, model_size: str, seed: int) -> Path:
    return Path(artifact_root) / 'runs' / run_name / model_size / f'seed{seed}'


def baseline_dir_for(artifact_root: str, model_size: str) -> Path:
    """Base-model results are shared by every gate and seed of one model size."""
    return Path(artifact_root) / 'baseline' / model_size


def calibration_path(artifact_root: str, model_size: str) -> Path:
    """MEG partitioner choice made by scripts/calibrate_meg.py (pre-registered rule)."""
    return Path(artifact_root) / 'calibration' / model_size / 'calibration.json'
