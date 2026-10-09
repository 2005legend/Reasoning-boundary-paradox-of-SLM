"""Single-GPU GRPO(+gate) training for the HEG-GRPO ablation.

    python train.py --gate heg_grpo --model_size 0.5B --seed 0 --tier standard \
        --final_eval math500,gsm8k --s3_uri s3://my-bucket/heg-grpo --usd_per_hour 1.006

Re-running the same command resumes from the last checkpoint; a finished run skips
straight to any final evals that are still missing.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from rlvr.clustering import (cluster_embeddings, cluster_size_distribution, embed_texts,
                             load_cluster_assignments, save_cluster_assignments)
from rlvr.config import (CLUSTER_BASES, GATE_TYPES, MODEL_SIZES, PARTITION_METHODS, TIER_SETTINGS,
                         TRAIN_DATASETS, ClusterConfig, ExperimentConfig, GateConfig, GenerationConfig,
                         OptimConfig, baseline_dir_for, calibration_path)
from rlvr.data import load_benchmark, load_train_problems
from rlvr.eval_runner import problem_subset, run_benchmark_eval, run_eval_probe, score_pass_at_k
from rlvr.evaluation import check_ceiling_effect, compute_shrinkage_slope
from rlvr.gates import build_gate
from rlvr.grpo_core import PartitionSettings, run_training_step
from rlvr.model_utils import generation_copy, load_lora_model, load_tokenizer, report_vram
from rlvr.trainer_utils import PromptSampler, S3Syncer, find_checkpoint, load_checkpoint, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gate', choices=GATE_TYPES, required=True)
    p.add_argument('--model_size', choices=MODEL_SIZES, default='0.5B')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tier', choices=list(TIER_SETTINGS), default='standard')
    p.add_argument('--steps', type=int, default=None, help='override the tier step count')
    p.add_argument('--tag', default='', help='suffix for the run name, e.g. lr1e-5')
    p.add_argument('--lr', type=float, default=OptimConfig.learning_rate)
    p.add_argument('--rollouts_per_prompt', type=int, default=GenerationConfig.rollouts_per_prompt)
    p.add_argument('--prompts_per_micro_step', type=int, default=OptimConfig.prompts_per_micro_step)
    p.add_argument('--grad_accum_steps', type=int, default=OptimConfig.grad_accum_steps)
    p.add_argument('--max_new_tokens', type=int, default=GenerationConfig.max_new_tokens)
    p.add_argument('--score_chunk_size', type=int, default=OptimConfig.score_chunk_size)
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--no_merged_generation', action='store_true',
                   help='sample rollouts through the LoRA layers instead of a merged copy (slower)')
    p.add_argument('--kl_beta', type=float, default=0.0)
    p.add_argument('--train_dataset', choices=TRAIN_DATASETS, default='math')
    p.add_argument('--math_levels', default='1,2,3,4,5', help='comma list of MATH levels to train on')
    p.add_argument('--clustering_basis', choices=CLUSTER_BASES, default=None,
                   help="default: 'subject' for MATH, 'semantic' for GSM8K")
    p.add_argument('--combine_op', choices=['multiplicative', 'min', 'harmonic_mean'], default='multiplicative')
    p.add_argument('--self_lambda', type=float, default=GateConfig.self_lambda)
    p.add_argument('--meg_alpha', type=float, default=GateConfig.meg_alpha)
    p.add_argument('--meg_partition', choices=('auto',) + PARTITION_METHODS, default='auto',
                   help="'auto' = the partitioner chosen by scripts/calibrate_meg.py")
    p.add_argument('--meg_tau_mode', type=float, default=None)
    p.add_argument('--meg_bigram_threshold', type=float, default=None)
    p.add_argument('--meg_random_partition', action='store_true', help='random-partition control')
    p.add_argument('--dump_groups_every', type=int, default=0, help='write sampled groups every N steps (0 = off)')
    p.add_argument('--eval_probe', type=int, default=0,
                   help='after training, time N in-domain problems x n_samples to measure eval cost')
    p.add_argument('--final_eval', default='', help="comma list: math500,gsm8k,gsm8k_platinum ('none' = skip)")
    p.add_argument('--artifact_root', default=None)
    p.add_argument('--s3_uri', default=None)
    p.add_argument('--usd_per_hour', type=float, default=0.0, help='for live cost estimates')
    p.add_argument('--device', default=None, help='default: cuda if available else cpu')
    p.add_argument('--train_limit', type=int, default=None, help='debug: use only N train prompts')
    return p.parse_args()


def resolve_partition(args, artifact_root: str) -> dict:
    """Explicit flags win; otherwise use the calibrated choice; otherwise the config defaults."""
    chosen = {}
    cal = calibration_path(artifact_root, args.model_size)
    if args.meg_partition == 'auto' and cal.exists():
        chosen = json.loads(cal.read_text())['chosen']
    elif args.meg_partition == 'auto' and args.gate in ('meg', 'heg_grpo'):
        # The uncalibrated default can put every correct rollout in one mode (MEG becomes a no-op).
        raise SystemExit(f'{args.gate} needs a calibrated partitioner and {cal} does not exist: run the '
                         f'"calibrate {args.model_size}" queue line first, or pass --meg_partition embedding|bigram.')
    elif args.meg_partition != 'auto':
        chosen = {'meg_partition': args.meg_partition}
    if args.meg_tau_mode is not None:
        chosen['meg_tau_mode'] = args.meg_tau_mode
    if args.meg_bigram_threshold is not None:
        chosen['meg_bigram_threshold'] = args.meg_bigram_threshold
    return {k: chosen[k] for k in ('meg_partition', 'meg_tau_mode', 'meg_bigram_threshold') if k in chosen}


def build_config(args) -> ExperimentConfig:
    kw = {}
    if args.artifact_root:
        kw['artifact_root'] = args.artifact_root
    artifact_root = kw.get('artifact_root', ExperimentConfig.__dataclass_fields__['artifact_root'].default)
    basis = args.clustering_basis or ('subject' if args.train_dataset == 'math' else 'semantic')
    return ExperimentConfig(
        model_size=args.model_size, compute_tier=args.tier, seed=args.seed, tag=args.tag,
        steps_override=args.steps, train_dataset=args.train_dataset,
        math_levels=[int(x) for x in args.math_levels.split(',') if x],
        gate=GateConfig(gate_type=args.gate, combine_op=args.combine_op, self_lambda=args.self_lambda,
                        meg_alpha=args.meg_alpha, meg_random_partition=args.meg_random_partition,
                        **resolve_partition(args, artifact_root)),
        generation=GenerationConfig(rollouts_per_prompt=args.rollouts_per_prompt,
                                    max_new_tokens=args.max_new_tokens,
                                    merged_generation=not args.no_merged_generation),
        optim=OptimConfig(learning_rate=args.lr, kl_beta=args.kl_beta,
                          prompts_per_micro_step=args.prompts_per_micro_step,
                          grad_accum_steps=args.grad_accum_steps,
                          score_chunk_size=args.score_chunk_size,
                          gradient_checkpointing=args.gradient_checkpointing),
        cluster=ClusterConfig(basis=basis),
        **kw,
    )


def fmt_hms(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    return f'{seconds // 3600:02d}h{(seconds % 3600) // 60:02d}m'


def get_clusters(cfg: ExperimentConfig, train_problems: list, embed_device: str, log) -> tuple:
    """Macro topic id per training prompt, and the number of topics."""
    if cfg.cluster.basis == 'subject':
        subjects = sorted({p.subject for p in train_problems})
        index = {s: i for i, s in enumerate(subjects)}
        labels = np.array([index[p.subject] for p in train_problems], dtype=np.int64)
        log(f'Macro topics = MATH subjects: {dict(zip(subjects, np.bincount(labels).tolist()))}')
        return labels, len(subjects)
    cache = Path(cfg.artifact_root) / 'clusters' / (
        f'{cfg.train_dataset}_{cfg.model_size}_{cfg.cluster.basis}_k{cfg.cluster.n_clusters}'
        f'_n{len(train_problems)}.json')
    if not cache.exists():
        log('Computing prompt embeddings + KMeans clusters (one-time)...')
        emb = embed_texts([p.prompt_text for p in train_problems], cfg.cluster.embedding_model,
                          device=embed_device, show_progress=True)
        solve_rates = None
        if cfg.cluster.basis == 'difficulty_aware':
            sr_path = Path(cfg.artifact_root) / 'solve_rates.json'
            if not sr_path.exists():
                raise RuntimeError('difficulty_aware clustering needs solve_rates.json (Req 43.1).')
            solve_rates = np.array(json.loads(sr_path.read_text())['solve_rates'])
        labels = cluster_embeddings(emb, cfg.cluster.n_clusters, basis=cfg.cluster.basis,
                                    solve_rates=solve_rates, random_state=cfg.cluster.random_state)
        save_cluster_assignments(labels, [p.prompt_id for p in train_problems], str(cache))
        log(f'Cluster sizes: {cluster_size_distribution(labels, cfg.cluster.n_clusters)}')
    prompt_ids, labels = load_cluster_assignments(str(cache))
    if prompt_ids != [p.prompt_id for p in train_problems]:
        raise RuntimeError(f'Cluster cache {cache} does not match the training prompts; delete it.')
    return labels, cfg.cluster.n_clusters


def monitor_settings(cfg: ExperimentConfig) -> dict:
    g = cfg.generation
    return {'benchmark': cfg.in_domain_benchmark,
            'n_problems': g.monitor_eval_problems, 'n_samples': g.monitor_eval_samples,
            'k_values': [1, 2, 3, 5, 10], 'temperature': cfg.eval.temperature, 'top_p': cfg.eval.top_p,
            'max_new_tokens': g.max_new_tokens, 'sample_seed': cfg.eval.sample_seed}


def log_merged_generation_gap(model, tokenizer, text: str, log) -> None:
    """Rollouts are sampled from a merged copy of the LoRA model (model_utils.generation_copy). Log how far
    the copy and the base model are from the LoRA model (mean |logit difference|). Information only: in bf16
    both gaps sit at the rounding-noise level until the adapters have moved well past it (a hard threshold
    stopped a healthy pilot run at step 5, 2026-10-06). Exact equivalence is tested in fp32 in
    tests/test_offline.py::test_generation_copy_matches_lora_and_leaves_model_untouched."""
    import torch
    ids = tokenizer(text, return_tensors='pt').input_ids[:, :96].to(next(model.parameters()).device)
    with torch.no_grad():
        lora = model(input_ids=ids).logits.float()
        with model.disable_adapter():
            base = model(input_ids=ids).logits.float()
        with generation_copy(model) as gen_model:
            merged = gen_model(input_ids=ids).logits.float()
    log(f'merged-generation gap (mean |dlogit|): copy-vs-LoRA {float((merged - lora).abs().mean()):.2e}, '
        f'base-vs-LoRA {float((base - lora).abs().mean()):.2e}')


def monitor_eval(model, tokenizer, problems, s: dict, microbatch: int) -> dict:
    pk, _, _, _ = score_pass_at_k(model, tokenizer, problems, s['n_samples'], s['k_values'],
                                  s['temperature'], s['top_p'], s['max_new_tokens'], microbatch,
                                  s['sample_seed'])
    return pk


def main():
    args = parse_args()
    import torch
    from torch.optim import AdamW
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

    cfg = build_config(args)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    embed_device = device
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = open(run_dir / 'stdout.log', 'a', buffering=1)

    def log(msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        stdout_log.write(line + '\n')

    s3 = S3Syncer(args.s3_uri, log)
    remote_suffix = run_dir.relative_to(cfg.artifact_root).as_posix()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg_path = run_dir / 'config.json'
    if cfg_path.exists():
        prior = json.loads(cfg_path.read_text())
        prior.pop('artifact_root', None)
        now = json.loads(cfg.to_json())
        now.pop('artifact_root', None)
        if prior != now:
            raise SystemExit(f'{run_dir} already holds a run with a different config. '
                             f'Use a new --tag, or delete that directory.')
    cfg.to_json(str(cfg_path))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    log(f'run={cfg.run_name} size={cfg.model_size} seed={cfg.seed} tier={cfg.compute_tier} '
        f'steps={cfg.training_steps} device={device}'
        + (f' gpu={torch.cuda.get_device_name(0)}' if device.startswith('cuda') else ''))

    tokenizer = load_tokenizer(cfg.model_name)
    done_path = run_dir / 'DONE.json'
    final_benchmarks = [b for b in args.final_eval.split(',') if b and b != 'none']

    if not done_path.exists():
        model = load_lora_model(cfg.model_name, device, cfg.optim.lora_r, cfg.optim.lora_alpha,
                                cfg.optim.lora_dropout, cfg.optim.gradient_checkpointing)
        log(report_vram('after model load'))

        train_problems = load_train_problems(cfg.train_dataset, tokenizer, cfg.math_levels)
        if args.train_limit:
            train_problems = train_problems[:args.train_limit]
        eval_problems = problem_subset(load_benchmark(cfg.in_domain_benchmark, tokenizer),
                                       cfg.generation.monitor_eval_problems, cfg.eval.subset_seed)
        prompts_per_update = cfg.optim.prompts_per_micro_step * cfg.optim.grad_accum_steps
        epochs = cfg.training_steps * prompts_per_update / len(train_problems)
        log(f'{len(train_problems)} {cfg.train_dataset} train prompts, {len(eval_problems)} '
            f'{cfg.in_domain_benchmark} monitor problems; {epochs:.2f} epochs = '
            f'{epochs * cfg.generation.rollouts_per_prompt:.1f} rollouts per training problem')

        cluster_labels, n_topics = get_clusters(cfg, train_problems, embed_device, log)
        records = [{'prompt_id': i, 'cluster_id': int(cluster_labels[i]),
                    'chat_prompt': p.chat_prompt, 'ground_truth': p.ground_truth}
                   for i, p in enumerate(train_problems)]

        gate = build_gate(cfg.gate.gate_type, n_topics, cfg.gate)
        partition = PartitionSettings(method=cfg.gate.meg_partition,
                                      randomize_for_gate=cfg.gate.meg_random_partition,
                                      tau_mode=cfg.gate.meg_tau_mode,
                                      bigram_threshold=cfg.gate.meg_bigram_threshold,
                                      embedding_model=cfg.gate.meg_embedding_model, device=embed_device)
        log(f'Gate: {gate.name} | mode partition: {partition.method}'
            + (' (randomized for the gate: control run)' if partition.randomize_for_gate else ''))

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=cfg.optim.learning_rate)
        warmup = int(cfg.optim.warmup_ratio * cfg.training_steps)
        scheduler = (get_cosine_schedule_with_warmup(optimizer, warmup, cfg.training_steps)
                     if cfg.optim.lr_scheduler_type == 'cosine'
                     else get_constant_schedule_with_warmup(optimizer, warmup))
        sampler = PromptSampler(len(records), cfg.optim.prompts_per_micro_step, cfg.seed)

        ckpt_dir = run_dir / 'checkpoint'
        start_step = 0
        found = find_checkpoint(ckpt_dir)
        if found is not None:
            start_step = load_checkpoint(found, model, optimizer, scheduler, gate)
            log(f'Resumed from {found.name} at step {start_step}')

        # Monitor baseline: base model on the same monitor problems/settings, shared by all gates/seeds.
        mon = monitor_settings(cfg)
        base_path = baseline_dir_for(cfg.artifact_root, cfg.model_size) / 'monitor_baseline.json'
        base_pk = None
        if base_path.exists():
            saved = json.loads(base_path.read_text())
            if saved.get('settings') == mon:
                base_pk = saved['pass_at_k']
        if base_pk is None and cfg.eval_interval > 0:
            log('Computing base-model monitor Pass@k (one-time per model size)...')
            with model.disable_adapter():
                base_pk = monitor_eval(model, tokenizer, eval_problems, mon,
                                       cfg.generation.monitor_eval_microbatch_prompts)
            base_path.parent.mkdir(parents=True, exist_ok=True)
            base_path.write_text(json.dumps({'settings': mon, 'pass_at_k': base_pk,
                                             'ceiling_flags': check_ceiling_effect(base_pk)}, indent=2))
            log(f'Base monitor Pass@k: {base_pk}')

        log_path = run_dir / 'train_log.jsonl'
        t_start = time.time()
        recent: list = []
        log(f'Training {cfg.training_steps} steps: {cfg.optim.prompts_per_micro_step} prompts x '
            f'{cfg.generation.rollouts_per_prompt} rollouts x {cfg.optim.grad_accum_steps} micro-steps '
            f'per update, lr={cfg.optim.learning_rate:g}')
        for step in range(start_step, cfg.training_steps):
            t_step = time.time()
            metrics = []
            for micro in range(cfg.optim.grad_accum_steps):
                idx = sampler.get_batch(step * cfg.optim.grad_accum_steps + micro)
                metrics.append(run_training_step(
                    model, tokenizer, gate, [records[i] for i in idx],
                    group_size=cfg.generation.rollouts_per_prompt,
                    max_new_tokens=cfg.generation.max_new_tokens,
                    temperature=cfg.generation.temperature, top_p=cfg.generation.top_p,
                    top_k=cfg.generation.top_k, repetition_penalty=cfg.generation.repetition_penalty,
                    clip_eps=cfg.optim.clip_eps, kl_beta=cfg.optim.kl_beta,
                    reward_mode=cfg.reward.reward_mode, format_weight=cfg.reward.format_weight,
                    correctness_weight=cfg.reward.correctness_weight,
                    score_chunk_size=cfg.optim.score_chunk_size,
                    loss_scale=1.0 / cfg.optim.grad_accum_steps, partition=partition,
                    rng_seed=(cfg.seed, step, micro), merged_generation=cfg.generation.merged_generation,
                    collect_groups=bool(args.dump_groups_every) and step % args.dump_groups_every == 0))
            if metrics[0].groups:
                with open(run_dir / 'groups.jsonl', 'a') as f:
                    for m in metrics:
                        for g in m.groups:
                            f.write(json.dumps({'step': step, **g}) + '\n')
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.optim.max_grad_norm))
            lr_used = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step_seconds = time.time() - t_step
            recent = (recent + [step_seconds])[-20:]
            if cfg.generation.merged_generation and (step + 1) % 25 == 0 and step < 100:
                log_merged_generation_gap(model, tokenizer, records[0]['chat_prompt'], log)

            if step % cfg.log_interval == 0 or step == cfg.training_steps - 1:
                def mean(attr):
                    vals = np.array([getattr(m, attr) for m in metrics], dtype=np.float64)
                    return None if np.isnan(vals).all() else float(np.nanmean(vals))
                rec = {'step': step, 'step_seconds': step_seconds, 'lr': lr_used, 'grad_norm': grad_norm}
                for attr in ('loss', 'mean_reward', 'format_success_rate', 'correctness_success_rate',
                             'mean_gate_weight', 'mean_pos_weight', 'frac_gated_below_1', 'approx_kl',
                             'mean_completion_tokens', 'frac_truncated', 'frac_zero_var_groups'):
                    rec[attr] = mean(attr)
                for k in metrics[0].timing:
                    rec[k] = float(np.sum([m.timing[k] for m in metrics]))
                for k in sorted({k for m in metrics for k in m.extra}):
                    rec[k] = float(np.mean([m.extra[k] for m in metrics if k in m.extra]))
                if hasattr(gate, 'diagnostic_snapshot'):
                    rec['gate_diagnostic'] = gate.diagnostic_snapshot()
                with open(log_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                sec_per_step = float(np.median(recent))
                eta = sec_per_step * (cfg.training_steps - step - 1)
                cost = (f' | ${args.usd_per_hour * (time.time() - t_start) / 3600:.2f} so far,'
                        f' ~${args.usd_per_hour * eta / 3600:.2f} left') if args.usd_per_hour else ''
                ent = f" ent={rec['mean_group_mode_entropy']:.3f}" if 'mean_group_mode_entropy' in rec else ''
                sel = f" sel={rec['frac_prompts_selected']:.2f}" if 'frac_prompts_selected' in rec else ''
                pos_w = f"{rec['mean_pos_weight']:.3f}" if rec['mean_pos_weight'] is not None else 'n/a'
                log(f"step {step:4d}/{cfg.training_steps} | {step_seconds:5.1f}s | "
                    f"reward={rec['mean_reward']:+.3f} cor={rec['correctness_success_rate']:.2f} "
                    f"zv={rec['frac_zero_var_groups']:.2f} | gate_w={rec['mean_gate_weight']:.3f} "
                    f"pos_w={pos_w}{ent}{sel} | kl={rec['approx_kl']:.4f} | "
                    f"len={rec['mean_completion_tokens']:.0f} trunc={rec['frac_truncated']:.2f} | "
                    f"lr={rec['lr']:.2e} | ETA {fmt_hms(eta)}{cost}")

            is_last = step == cfg.training_steps - 1
            if (step + 1) % cfg.checkpoint_interval == 0 or is_last:
                save_checkpoint(ckpt_dir, model, optimizer, scheduler, gate, step + 1)
                log(f'checkpoint saved at step {step + 1}')
                s3.sync(run_dir, remote_suffix, wait=is_last)

            if cfg.eval_interval > 0 and base_pk is not None and (step + 1) % cfg.eval_interval == 0:
                with generation_copy(model, cfg.generation.merged_generation) as gen_model:
                    pk = monitor_eval(gen_model, tokenizer, eval_problems, mon,
                                      cfg.generation.monitor_eval_microbatch_prompts)
                slope = compute_shrinkage_slope(base_pk, pk)
                with open(run_dir / 'eval_log.jsonl', 'a') as f:
                    f.write(json.dumps({'step': step + 1, 'pass_at_k': pk, 'shrinkage_slope': slope.slope,
                                        'delta_pass_k': slope.delta_pass_k}) + '\n')
                log(f'[monitor eval @ {step + 1}] Pass@k={pk} slope={slope.slope:+.4f}')

        train_seconds = time.time() - t_start
        model.save_pretrained(str(run_dir / 'adapter_final'))
        done_path.write_text(json.dumps({
            'steps': cfg.training_steps, 'resumed_from': start_step, 'train_seconds': train_seconds,
            'median_step_seconds': float(np.median(recent)) if recent else None,
            'usd_estimate_this_session': args.usd_per_hour * train_seconds / 3600,
            'n_train_problems': len(train_problems), 'rollouts_per_prompt': cfg.generation.rollouts_per_prompt,
            'epochs': epochs, 'rollouts_per_training_problem': epochs * cfg.generation.rollouts_per_prompt,
        }, indent=2))
        log(f'Training complete in {fmt_hms(train_seconds)} -> {run_dir}')
        del model, optimizer
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    probe_path = run_dir / 'eval_probe.json'
    if final_benchmarks or (args.eval_probe and not probe_path.exists()):
        from rlvr.model_utils import load_adapter_for_inference
        model = load_adapter_for_inference(cfg.model_name, str(run_dir / 'adapter_final'), device)
        if args.eval_probe and not probe_path.exists():
            run_eval_probe(model, tokenizer, cfg.in_domain_benchmark, cfg.eval, args.eval_probe, probe_path, log)
        for bench in final_benchmarks:
            run_benchmark_eval(model, tokenizer, bench, cfg.eval, run_dir / 'eval', log)
            s3.sync(run_dir, remote_suffix, wait=True)
    s3.sync(run_dir, remote_suffix, wait=True)
    stdout_log.close()


if __name__ == '__main__':
    main()
