"""GRPO advantages, loss, and one gated training micro-step (Req 6, 7, 39, 44). From scratch, not TRL."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .gates import Gate, GateContext, compute_group_partitions, identity_sync, randomize_partition
from .model_utils import build_scoring_batch, generate_rollouts, generation_copy, token_logprobs
from .rewards import answers_equivalent, compute_reward, parse_output


@dataclass
class PartitionSettings:
    method: str = 'embedding'          # real partitioner: 'embedding' | 'bigram'
    randomize_for_gate: bool = False   # random-partition control (diagnostics still use the real one)
    tau_mode: float = 0.85
    bigram_threshold: float = 0.5
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    device: str = 'cpu'


@dataclass
class StepMetrics:
    loss: float
    mean_reward: float
    format_success_rate: float
    correctness_success_rate: float
    mean_gate_weight: float
    mean_pos_weight: float
    frac_gated_below_1: float
    approx_kl: float
    mean_completion_tokens: float
    frac_truncated: float
    frac_zero_var_groups: float
    timing: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    groups: list = field(default_factory=list)


def compute_group_advantages(rewards, group_size: int):
    """GRPO group-relative advantage. A zero-variance group gets all-zero advantage."""
    import torch
    n_prompts = rewards.shape[0] // group_size
    r = rewards.view(n_prompts, group_size)
    mu = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, unbiased=True, keepdim=True) if group_size > 1 else torch.ones_like(mu)
    return ((r - mu) / (std + 1e-4)).view(-1)


def ppo_clip_token_loss(new_logprobs, old_logprobs, advantages, clip_eps: float):
    """Per-token PPO-clip objective (arXiv:2510.02230 Eq. 11), unreduced."""
    import torch
    ratio = torch.exp(new_logprobs - old_logprobs)
    adv = advantages.unsqueeze(-1)
    return -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)


def k3_kl(new_logprobs, ref_logprobs):
    """Per-token k3 estimator of KL(pi || ref)."""
    import torch
    diff = ref_logprobs - new_logprobs
    return torch.exp(diff) - diff - 1.0


def run_training_step(model, tokenizer, gate: Gate, prompt_records: list, *, group_size: int,
                      max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                      repetition_penalty: float, clip_eps: float, kl_beta: float,
                      reward_mode: str, format_weight: float, correctness_weight: float,
                      score_chunk_size: int, loss_scale: float, partition: PartitionSettings,
                      rng_seed=(0,), collect_groups: bool = False, merged_generation: bool = False,
                      sync_fn=identity_sync) -> StepMetrics:
    """One on-policy GRPO(+gate) micro-step; gradients are accumulated into the LoRA params.

    The loss is the token-level mean over every completion token of this micro-step, times
    loss_scale (= 1/grad_accum_steps). It is evaluated chunk by chunk with that same global
    denominator, so summing the chunk gradients gives exactly the full-batch gradient while
    only one chunk's activations are alive at a time.

    With one update per rollout batch and no dropout, old-policy log-probs equal the current
    ones, so old = new.detach() (ratio == 1, gradient == -A * grad log pi) and no extra forward
    pass is needed.
    """
    import torch
    timing = {}
    device = next(model.parameters()).device
    need_greedy = bool(getattr(gate, 'requires_greedy', False))

    t0 = time.perf_counter()
    with generation_copy(model, merged_generation) as gen_model:
        rollouts = generate_rollouts(gen_model, tokenizer, prompt_records, group_size, max_new_tokens,
                                     temperature, top_p, top_k, repetition_penalty,
                                     microbatch_prompts=len(prompt_records), greedy_extra=need_greedy)
    timing['generate_s'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    breakdowns = [compute_reward(c, gt, reward_mode, format_weight, correctness_weight)
                  for c, gt in zip(rollouts.completions, rollouts.ground_truths)]
    rewards_t = torch.tensor([b.total for b in breakdowns], dtype=torch.float32, device=device)
    is_correct = np.array([1.0 if b.is_correct else 0.0 for b in breakdowns])
    advantages = compute_group_advantages(rewards_t, group_size)
    grouped = rewards_t.view(-1, group_size)
    frac_zero_var = float((grouped.max(dim=1).values == grouped.min(dim=1).values).float().mean())
    greedy_correct = None
    if need_greedy:
        per_prompt = [answers_equivalent(parse_output(c).boxed, r['ground_truth'])
                      for c, r in zip(rollouts.greedy_completions, prompt_records)]
        greedy_correct = np.repeat(np.array(per_prompt, dtype=bool), group_size)
    timing['reward_s'] = time.perf_counter() - t0

    # Real solution-mode partition: drives MEG and the mode-entropy diagnostic of every condition.
    t0 = time.perf_counter()
    labels, part_diag = compute_group_partitions(
        rollouts.completions, is_correct, group_size, method=partition.method,
        tau_mode=partition.tau_mode, bigram_threshold=partition.bigram_threshold,
        embedding_model=partition.embedding_model, device=partition.device)
    gate_labels = labels
    if partition.randomize_for_gate:
        gate_labels = randomize_partition(labels, group_size, np.random.default_rng(list(rng_seed)))
    timing['partition_s'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ctx = GateContext.build(rollouts.cluster_ids, rollouts.prompt_ids, is_correct, group_size,
                            partition=gate_labels, greedy_correct=greedy_correct)
    adv_np = advantages.detach().cpu().numpy()
    gw_np = gate.step(ctx, adv_np, sync_fn=sync_fn)
    gated_adv = advantages * torch.tensor(gw_np, dtype=torch.float32, device=device)
    timing['gate_s'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.train()
    n_seq = len(rollouts.completion_token_ids)
    total_tokens = float(max(sum(len(c) for c in rollouts.completion_token_ids), 1))
    loss_total, kl_sum, kl_tokens = 0.0, 0.0, 0.0
    for s in range(0, n_seq, score_chunk_size):
        e = min(s + score_chunk_size, n_seq)
        ids, attn, n_keep, tgt, mask = build_scoring_batch(
            rollouts.prompt_token_ids[s:e], rollouts.completion_token_ids[s:e],
            tokenizer.pad_token_id, device)
        new_lp = token_logprobs(model, ids, attn, n_keep, tgt)
        per_tok = ppo_clip_token_loss(new_lp, new_lp.detach(), gated_adv[s:e], clip_eps)
        chunk_loss = (per_tok * mask).sum() / total_tokens

        # The reference KL is always measured on the first chunk (cheap progress signal:
        # is the policy actually moving?) and on every chunk when it is part of the loss.
        if kl_beta > 0 or s == 0:
            with torch.no_grad(), model.disable_adapter():
                ref_lp = token_logprobs(model, ids, attn, n_keep, tgt)
            kl_tok = k3_kl(new_lp, ref_lp)
            kl_sum += float((kl_tok.detach() * mask).sum())
            kl_tokens += float(mask.sum())
            if kl_beta > 0:
                chunk_loss = chunk_loss + kl_beta * (kl_tok * mask).sum() / total_tokens

        (chunk_loss * loss_scale).backward()
        loss_total += float(chunk_loss.detach())
        del ids, attn, tgt, mask, new_lp, per_tok, chunk_loss
    timing['backward_s'] = time.perf_counter() - t0

    extra = dict(part_diag)
    if greedy_correct is not None:
        extra['frac_prompts_selected'] = float(1.0 - greedy_correct[::group_size].mean())
    groups = []
    if collect_groups:
        for g, start in enumerate(range(0, n_seq, group_size)):
            sl = slice(start, start + group_size)
            groups.append({'prompt_id': int(rollouts.prompt_ids[start]),
                           'cluster_id': int(rollouts.cluster_ids[start]),
                           'ground_truth': rollouts.ground_truths[start],
                           'completions': rollouts.completions[sl],
                           'is_correct': is_correct[sl].astype(int).tolist(),
                           'mode_labels': labels[sl].tolist(), 'gate_labels': gate_labels[sl].tolist(),
                           'advantages': adv_np[sl].round(4).tolist(), 'gate_weights': gw_np[sl].round(4).tolist(),
                           'greedy_correct': None if greedy_correct is None else bool(greedy_correct[start])})
    pos = adv_np > 0
    return StepMetrics(
        loss=loss_total,
        mean_reward=float(rewards_t.mean()),
        format_success_rate=float(np.mean([b.is_well_formed for b in breakdowns])),
        correctness_success_rate=float(is_correct.mean()),
        mean_gate_weight=float(gw_np.mean()),
        mean_pos_weight=float(gw_np[pos].mean()) if pos.any() else float('nan'),
        frac_gated_below_1=float((gw_np < 0.999).mean()),
        approx_kl=kl_sum / max(kl_tokens, 1.0),
        mean_completion_tokens=float(np.mean([len(c) for c in rollouts.completion_token_ids])),
        frac_truncated=float(np.mean(rollouts.truncated)),
        frac_zero_var_groups=frac_zero_var,
        timing=timing,
        extra=extra,
        groups=groups,
    )
