"""Pass@k evaluation shared by train.py (monitor + final) and evaluate.py (base model / re-scoring)."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from .config import EvalConfig
from .data import load_benchmark
from .evaluation import check_ceiling_effect, compute_pass_at_k
from .model_utils import generate_rollouts
from .rewards import answers_equivalent, parse_output


def problem_subset(problems: list, n_problems: int, subset_seed: int) -> list:
    """Deterministic subset, identical for every model, so base/run comparisons are paired."""
    if n_problems >= len(problems):
        return list(problems)
    idx = np.sort(np.random.default_rng(subset_seed).permutation(len(problems))[:n_problems])
    return [problems[i] for i in idx]


def completions_path(json_path: Path) -> Path:
    return json_path.with_name(json_path.stem + '_completions.jsonl.gz')


def write_completions(path: Path, problems: list, completions: list, correctness: np.ndarray) -> None:
    """Every sampled completion, one line per problem. Needed for the blinded human audit
    (were the base model's successes on problems RLVR loses valid reasoning or lucky guesses?)."""
    n = correctness.shape[1]
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for i, p in enumerate(problems):
            f.write(json.dumps({'prompt_id': p.prompt_id, 'problem': p.prompt_text, 'ground_truth': p.ground_truth,
                                'correct': correctness[i].tolist(),
                                'completions': completions[i * n:(i + 1) * n]}) + '\n')


def read_completions(path: Path) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return {r['prompt_id']: r for r in map(json.loads, f)}


def score_pass_at_k(model, tokenizer, problems: list, n_samples: int, k_values: list,
                    temperature: float, top_p: float, max_new_tokens: int,
                    microbatch_prompts: int, sample_seed: int, return_completions: bool = False) -> tuple:
    """Returns (pass_at_k dict, correctness matrix [P, n], mean completion tokens, frac truncated),
    plus the completions (problem-major) when return_completions is set."""
    import torch
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(sample_seed)
    try:
        records = [{'prompt_id': i, 'chat_prompt': p.chat_prompt, 'ground_truth': p.ground_truth}
                   for i, p in enumerate(problems)]
        r = generate_rollouts(model, tokenizer, records, n_samples, max_new_tokens, temperature,
                              top_p, microbatch_prompts=microbatch_prompts)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    correctness = np.array([1 if answers_equivalent(parse_output(c).boxed, gt) else 0
                            for c, gt in zip(r.completions, r.ground_truths)],
                           dtype=np.int8).reshape(len(problems), n_samples)
    mean_tokens = float(np.mean([len(c) for c in r.completion_token_ids]))
    out = (compute_pass_at_k(correctness, k_values), correctness, mean_tokens, float(np.mean(r.truncated)))
    return out + (list(r.completions),) if return_completions else out


def run_eval_probe(model, tokenizer, benchmark: str, eval_cfg: EvalConfig, n_problems: int,
                   out_path: Path, log=print) -> dict:
    """Time n_problems x n_samples with the final-eval settings, so the pilot can budget evals."""
    settings = eval_cfg.settings_for(benchmark)
    probe = problem_subset(load_benchmark(benchmark, tokenizer), n_problems, settings['subset_seed'])
    t0 = time.time()
    score_pass_at_k(model, tokenizer, probe, settings['n_samples'], settings['k_values'],
                    settings['temperature'], settings['top_p'], settings['max_new_tokens'],
                    eval_cfg.microbatch_prompts, settings['sample_seed'])
    secs = time.time() - t0
    result = {'benchmark': benchmark, 'problems': len(probe), 'n_samples': settings['n_samples'],
              'max_new_tokens': settings['max_new_tokens'], 'seconds': secs,
              'seconds_per_problem': secs / len(probe)}
    out_path.write_text(json.dumps(result, indent=2))
    log(f'[eval probe] {len(probe)} {benchmark} problems x {settings["n_samples"]} samples in '
        f'{secs / 60:.1f} min ({secs / len(probe):.1f} s/problem)')
    return result


def run_benchmark_eval(model, tokenizer, benchmark: str, eval_cfg: EvalConfig, out_dir: Path,
                       log=print) -> dict:
    """Full Pass@k eval on one benchmark; skips if a result with identical settings exists."""
    settings = eval_cfg.settings_for(benchmark)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{benchmark}_n{settings['n_samples']}_p{settings['n_problems']}"
    json_path, npz_path = out_dir / f'{stem}.json', out_dir / f'{stem}.npz'
    if json_path.exists():
        prior = json.loads(json_path.read_text())
        if prior.get('settings') == settings:
            log(f'[eval] {benchmark}: already done -> {json_path}')
            return prior
        log(f'[eval] {benchmark}: settings changed, re-running')

    problems = problem_subset(load_benchmark(benchmark, tokenizer), settings['n_problems'],
                              settings['subset_seed'])
    log(f'[eval] {benchmark}: {len(problems)} problems x {settings["n_samples"]} samples')
    t0 = time.time()
    pk, correctness, mean_tokens, frac_trunc, completions = score_pass_at_k(
        model, tokenizer, problems, settings['n_samples'], settings['k_values'],
        settings['temperature'], settings['top_p'], settings['max_new_tokens'],
        eval_cfg.microbatch_prompts, settings['sample_seed'], return_completions=True)
    write_completions(completions_path(json_path), problems, completions, correctness)
    result = {
        'settings': settings, 'pass_at_k': pk, 'ceiling_flags': check_ceiling_effect(pk),
        'mean_completion_tokens': mean_tokens, 'frac_truncated': frac_trunc,
        'wall_seconds': time.time() - t0,
    }
    np.savez_compressed(npz_path, correctness=correctness,
                        prompt_ids=np.array([p.prompt_id for p in problems]))
    json_path.write_text(json.dumps(result, indent=2))
    log(f'[eval] {benchmark}: Pass@k={pk} ({result["wall_seconds"] / 60:.1f} min)')
    return result
