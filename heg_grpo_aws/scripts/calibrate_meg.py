"""Choose MEG's solution-mode partitioner on BASE-model rollouts, before any training.

    python scripts/calibrate_meg.py --model_size 0.5B

Samples G rollouts for N MATH training prompts, clusters the correct ones with every candidate
(embedding cosine at several tau, word-bigram Jaccard at several thresholds) and applies the
pre-registered rule (PREREGISTRATION.md): among candidates whose median number of modes, over
groups with >= 4 correct rollouts, lies in [2, 4], take the one closest to 3; ties go to bigram
(arXiv:2606.29985: bigram distance tracks approach-level diversity better than embeddings), then to
the larger share of multi-mode groups. Writes calibration.json (read by train.py when
--meg_partition auto) and calibration_groups.jsonl for a manual audit of what the modes capture.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvr.config import DEFAULT_ARTIFACT_ROOT, MODEL_NAME_BY_SIZE, GenerationConfig, calibration_path
from rlvr.data import load_math_train
from rlvr.gates import compute_group_partitions
from rlvr.model_utils import generate_rollouts, load_adapter_for_inference, load_tokenizer
from rlvr.rewards import compute_reward


def group_mode_counts(labels: np.ndarray, group_size: int, min_correct: int) -> list:
    counts = []
    for start in range(0, len(labels), group_size):
        lab = labels[start:start + group_size]
        lab = lab[lab >= 0]
        if len(lab) >= min_correct:
            counts.append(len(np.unique(lab)))
    return counts


def choose(candidates: list) -> dict:
    valid = [c for c in candidates if c['median_modes'] is not None and 2 <= c['median_modes'] <= 4]
    pool = valid or [c for c in candidates if c['median_modes'] is not None]
    if not pool:
        return {}
    best = min(pool, key=lambda c: (abs(c['median_modes'] - 3), c['method'] != 'bigram', -c['frac_multi_mode']))
    return {**best, 'valid': bool(valid)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_size', default='0.5B')
    p.add_argument('--n_prompts', type=int, default=96)
    p.add_argument('--levels', default='1,2,3,4,5')
    p.add_argument('--max_new_tokens', type=int, default=GenerationConfig.max_new_tokens)
    p.add_argument('--microbatch_prompts', type=int, default=8)
    p.add_argument('--taus', default='0.80,0.85,0.90,0.95')
    p.add_argument('--bigram_thresholds', default='0.3,0.4,0.5,0.6,0.7')
    p.add_argument('--min_correct', type=int, default=4)
    p.add_argument('--artifact_root', default=DEFAULT_ARTIFACT_ROOT)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    gen = GenerationConfig()
    name = MODEL_NAME_BY_SIZE[args.model_size]
    tokenizer = load_tokenizer(name)
    model = load_adapter_for_inference(name, None, device)
    problems = load_math_train(tokenizer, [int(x) for x in args.levels.split(',')])
    pick = np.random.default_rng(args.seed).choice(len(problems), size=min(args.n_prompts, len(problems)),
                                                   replace=False)
    records = [{'prompt_id': int(i), 'chat_prompt': problems[i].chat_prompt,
                'ground_truth': problems[i].ground_truth} for i in pick]

    t0 = time.time()
    r = generate_rollouts(model, tokenizer, records, gen.rollouts_per_prompt, args.max_new_tokens,
                          gen.temperature, gen.top_p, gen.top_k, gen.repetition_penalty,
                          microbatch_prompts=args.microbatch_prompts)
    is_correct = np.array([compute_reward(c, gt).is_correct for c, gt in zip(r.completions, r.ground_truths)],
                          dtype=float)
    G = gen.rollouts_per_prompt
    print(f'{len(records)} prompts x {G} rollouts in {(time.time() - t0) / 60:.1f} min; '
          f'base correctness {is_correct.mean():.3f}')

    candidates = []
    for method, values, key in (('embedding', args.taus, 'meg_tau_mode'),
                                ('bigram', args.bigram_thresholds, 'meg_bigram_threshold')):
        for v in [float(x) for x in values.split(',')]:
            kw = {'tau_mode': v} if method == 'embedding' else {'bigram_threshold': v}
            labels, diag = compute_group_partitions(r.completions, is_correct, G, method=method,
                                                    device=device, **kw)
            counts = group_mode_counts(labels, G, args.min_correct)
            candidates.append({'method': method, key: v, 'groups_used': len(counts),
                               'median_modes': float(np.median(counts)) if counts else None,
                               'frac_multi_mode': float(np.mean(np.array(counts) > 1)) if counts else 0.0,
                               'mean_group_mode_entropy': diag.get('mean_group_mode_entropy'),
                               '_labels': labels})
            print(f"  {method:9s} {key}={v:.2f}: groups(>={args.min_correct} correct)={len(counts):3d} "
                  f"median modes={candidates[-1]['median_modes']} multi-mode share={candidates[-1]['frac_multi_mode']:.2f}")

    best = choose(candidates)
    out_dir = calibration_path(args.artifact_root, args.model_size).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if not best:
        raise SystemExit('No group had enough correct rollouts; raise --n_prompts or use easier --levels.')
    chosen = {'meg_partition': best['method']}
    chosen.update({k: best[k] for k in ('meg_tau_mode', 'meg_bigram_threshold') if k in best})
    with open(out_dir / 'calibration_groups.jsonl', 'w') as f:
        for g, start in enumerate(range(0, len(r.completions), G)):
            f.write(json.dumps({'prompt_id': records[g]['prompt_id'], 'ground_truth': records[g]['ground_truth'],
                                'problem': problems[records[g]['prompt_id']].prompt_text,
                                'completions': r.completions[start:start + G],
                                'is_correct': is_correct[start:start + G].astype(int).tolist(),
                                'labels_chosen': best['_labels'][start:start + G].tolist()}) + '\n')
    for c in candidates:
        c.pop('_labels')
    best.pop('_labels', None)
    calibration_path(args.artifact_root, args.model_size).write_text(json.dumps({
        'chosen': chosen, 'valid': best['valid'], 'rule': 'median modes in [2,4], closest to 3; tie -> bigram',
        'base_correctness': float(is_correct.mean()), 'n_prompts': len(records), 'group_size': G,
        'candidates': candidates, 'created': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=2))
    flag = '' if best['valid'] else '  (WARNING: no candidate met the [2,4] rule; closest one taken)'
    print(f'\nChosen: {chosen}{flag}\nWrote {calibration_path(args.artifact_root, args.model_size)}')


if __name__ == '__main__':
    main()
