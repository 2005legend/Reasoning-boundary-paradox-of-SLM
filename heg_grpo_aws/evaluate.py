"""Final Pass@k evaluation for the base model or any finished run.

    python evaluate.py --base --model_size 0.5B --benchmarks math500,gsm8k
    python evaluate.py --run heg_grpo --model_size 0.5B --seed 0 --benchmarks math500

Settings come from rlvr.config.EvalConfig so base and trained models are always scored identically.
Results land in <run>/eval/ or <artifact_root>/baseline/<size>/eval/ and are skipped if present.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from rlvr.config import DEFAULT_ARTIFACT_ROOT, MODEL_NAME_BY_SIZE, MODEL_SIZES, EvalConfig, baseline_dir_for, run_dir_for
from rlvr.eval_runner import run_benchmark_eval
from rlvr.model_utils import load_adapter_for_inference, load_tokenizer
from rlvr.trainer_utils import S3Syncer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument('--base', action='store_true', help='evaluate the untrained base model')
    who.add_argument('--run', help='run name, e.g. heg_grpo or heg_grpo__lr1e-5')
    p.add_argument('--model_size', choices=MODEL_SIZES, default='0.5B')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--benchmarks', default='math500,gsm8k')
    p.add_argument('--artifact_root', default=DEFAULT_ARTIFACT_ROOT)
    p.add_argument('--s3_uri', default=None)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.base:
        out_root, adapter = baseline_dir_for(args.artifact_root, args.model_size), None
    else:
        out_root = run_dir_for(args.artifact_root, args.run, args.model_size, args.seed)
        adapter = out_root / 'adapter_final'
        if not adapter.exists():
            raise SystemExit(f'No finished adapter at {adapter} (is training complete?)')

    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    tokenizer = load_tokenizer(MODEL_NAME_BY_SIZE[args.model_size])
    model = load_adapter_for_inference(MODEL_NAME_BY_SIZE[args.model_size],
                                       str(adapter) if adapter else None, device)
    s3 = S3Syncer(args.s3_uri, log)
    eval_cfg = EvalConfig()
    for bench in [b for b in args.benchmarks.split(',') if b]:
        run_benchmark_eval(model, tokenizer, bench, eval_cfg, out_root / 'eval', log)
        s3.sync(out_root, Path(out_root).relative_to(args.artifact_root).as_posix(), wait=True)


if __name__ == '__main__':
    main()
