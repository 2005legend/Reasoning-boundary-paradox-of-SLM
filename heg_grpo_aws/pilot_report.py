"""Go/no-go report for a pilot run, plus a budget projection for the full ablation.

    python pilot_report.py --run heg_grpo__pilot_lr1e-5 --model_size 0.5B --seed 0 \
        --usd_per_hour 1.006 --budget 200 --steps 800 --core_runs 18 --extra_runs 6

The eval cost comes from the run's eval_probe.json (train.py --eval_probe) when present.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from rlvr.config import DEFAULT_ARTIFACT_ROOT, TIER_SETTINGS, EvalConfig, GenerationConfig, run_dir_for


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='heg_grpo')
    p.add_argument('--model_size', default='0.5B')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--artifact_root', default=DEFAULT_ARTIFACT_ROOT)
    p.add_argument('--usd_per_hour', type=float, default=1.006)
    p.add_argument('--budget', type=float, default=200.0)
    p.add_argument('--steps', type=int, default=800, help='planned steps per real run')
    p.add_argument('--core_runs', type=int, default=18, help='6 conditions x 3 seeds')
    p.add_argument('--extra_runs', type=int, default=6, help='random-partition control + BBG, 3 seeds each')
    p.add_argument('--gsm8k_cost_ratio', type=float, default=0.5,
                   help='GSM8K eval cost relative to MATH-500 per problem (shorter answers)')
    args = p.parse_args()

    run_dir = run_dir_for(args.artifact_root, args.run, args.model_size, args.seed)
    recs = [json.loads(l) for l in (run_dir / 'train_log.jsonl').read_text().splitlines() if l.strip()]
    if len(recs) < 3:
        raise SystemExit(f'Only {len(recs)} logged steps in {run_dir}; let the pilot run longer.')
    steady = recs[2:]  # first steps include CUDA warm-up / cache allocation
    sec = float(np.median([r['step_seconds'] for r in steady]))
    gate = args.run.split('__')[0]
    checks = []

    def check(ok: bool, label: str, detail: str, warn_only: bool = False):
        checks.append(('PASS' if ok else ('WARN' if warn_only else 'FAIL'), label, detail))

    def series(key):
        return [r[key] for r in recs if r.get(key) is not None]

    cor = series('correctness_success_rate')
    check(max(cor) > 0, 'reward is alive', f'correctness per step: first={cor[0]:.2f} max={max(cor):.2f} last={cor[-1]:.2f}')
    ent, multi = series('mean_group_mode_entropy'), series('frac_groups_multi_mode')
    check(len(ent) >= len(recs) // 2, 'mode diagnostics are logged',
          f'mean_group_mode_entropy on {len(ent)}/{len(recs)} steps' + (f', mean {np.mean(ent):.3f}' if ent else ''))
    if multi:
        check(np.mean(multi) > 0.1, 'partition finds more than one mode',
              f'mean share of groups with >1 correct mode = {np.mean(multi):.2f} (near 0 makes MEG a no-op)',
              warn_only=gate not in ('meg', 'heg_grpo'))
    if gate in ('meg', 'heg_grpo'):
        gw = series('frac_gated_below_1')
        check(max(gw) > 0, 'gate actually reweights something', f'max frac_gated_below_1={max(gw):.3f}')
    if gate in ('o_self', 'h_cb_grpo'):
        sel = series('frac_prompts_selected')
        ok = bool(sel) and 0.05 <= float(np.mean(sel)) <= 0.95
        check(ok, 'SELF filter is active but not total',
              f'share of prompts whose greedy answer fails = {np.mean(sel):.2f}' if sel else 'not logged')
    # approx_kl compares two bf16 forward passes (LoRA on / off). Until the LoRA delta outgrows bf16
    # rounding it sits on a noise floor of ~1e-4 to 3e-4 whatever the lr, so it only counts as
    # movement when clearly above that. The adapter norm (fp32, saved) grows with lr from step 1.
    kl = series('approx_kl')
    kl_tail = float(np.mean(kl[-5:]))
    check(kl_tail > 1e-3, 'policy has moved past the bf16 noise floor of the KL estimate',
          f'KL to base, mean of last 5 steps = {kl_tail:.2e} (~1e-4..3e-4 is rounding noise, not learning; '
          f'expected in a short pilot)', warn_only=True)
    adapter = run_dir / 'adapter_final' / 'adapter_model.safetensors'
    if adapter.exists():
        from safetensors.numpy import load_file
        b = [float(np.linalg.norm(v.astype(np.float64))) for k, v in load_file(str(adapter)).items() if 'lora_B' in k]
        check(bool(b) and max(b) > 0, 'LoRA update is non-zero',
              f'lora_B Frobenius norm: mean {np.mean(b):.2e}, max {max(b):.2e} (scales ~linearly with lr x steps)')
    trunc = float(np.mean(series('frac_truncated')))
    check(trunc < 0.10, 'completions fit in max_new_tokens', f'mean frac_truncated={trunc:.2f}', warn_only=True)
    zv = float(np.mean(series('frac_zero_var_groups')))
    check(zv < 0.8, 'groups carry learning signal',
          f'mean share of zero-variance groups={zv:.2f} (if high: --math_levels 1,2,3)', warn_only=True)

    print(f'\nPilot report: {run_dir}\n')
    for status, label, detail in checks:
        print(f'  [{status}] {label}: {detail}')

    timing_keys = [k for k in steady[0] if k.endswith('_s')]
    if timing_keys:
        tot = {k: float(np.median([r[k] for r in steady])) for k in timing_keys}
        s = sum(tot.values()) or 1.0
        print('\n  Where a step goes: ' + ', '.join(f"{k[:-2]} {100 * v / s:.0f}%" for k, v in tot.items()))

    probe_path = run_dir / 'eval_probe.json'
    n_math = EvalConfig().n_problems['math500']
    n_gsm = EvalConfig().n_problems['gsm8k']
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        eval_h = probe['seconds_per_problem'] * (n_math + args.gsm8k_cost_ratio * n_gsm) / 3600
        eval_src = f"measured: {probe['seconds_per_problem']:.1f} s/problem on {probe['benchmark']}"
    else:
        eval_h, eval_src = 1.5, 'GUESS (no eval_probe.json; run the pilot with --eval_probe 16)'
    # Main runs use the standard tier: a monitor eval (50 problems x 10 samples) every eval_interval
    # steps, plus one shared base-model monitor eval. Scaled from the probe by sample count.
    sec_per_problem = eval_h * 3600 / (n_math + args.gsm8k_cost_ratio * n_gsm)
    gen, every = GenerationConfig(), TIER_SETTINGS['standard']['eval_interval']
    mon_h = gen.monitor_eval_problems * gen.monitor_eval_samples / EvalConfig().n_samples * sec_per_problem / 3600
    n_mon = args.steps // every if every else 0
    hours_run = sec * args.steps / 3600 + n_mon * mon_h
    usd_run = hours_run * args.usd_per_hour
    usd_eval_model = eval_h * args.usd_per_hour

    def total(n_runs):
        return n_runs * (usd_run + usd_eval_model) + usd_eval_model + mon_h * args.usd_per_hour  # + base model

    print(f'\n  Median step: {sec:.1f}s  ->  {args.steps} steps + {n_mon} monitor evals ({n_mon * mon_h:.2f} h) '
          f'= {hours_run:.1f} h = ${usd_run:.2f} per run')
    print(f'  Final eval per model: {eval_h:.2f} h = ${usd_eval_model:.2f}  ({eval_src})')
    cap = args.budget * 0.85
    for label, n in (('core', args.core_runs), ('core + extras', args.core_runs + args.extra_runs)):
        t = total(n)
        verdict = 'fits' if t <= cap else 'OVER'
        print(f'  {label:14s} {n:2d} runs: ${t:7.2f}  ({verdict} the 85% line of ${cap:.0f})')
    if total(args.core_runs) > cap:
        per_run_budget = (cap - usd_eval_model) / args.core_runs - usd_eval_model
        sec_eff = sec + (mon_h * 3600 / every if every else 0.0)  # monitor evals amortized per step
        max_steps = int(per_run_budget / args.usd_per_hour * 3600 / sec_eff) if per_run_budget > 0 else 0
        print(f'  -> core runs fit at <= {max_steps} steps each, or drop seed 2 (seed-major queue).')
    if any(c[0] == 'FAIL' for c in checks):
        print('\n  NO-GO: fix the FAIL items before spending on real runs.')
    else:
        print('\n  GO (review any WARN items).')


if __name__ == '__main__':
    main()
