"""Prompt sampling, crash-safe checkpointing, and async S3 sync."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


class PromptSampler:
    """Deterministic per-epoch permutation; batch i depends only on (seed, i), so resume is exact."""

    def __init__(self, n_prompts: int, batch_size: int, seed: int):
        self.n = n_prompts
        self.batch_size = batch_size
        self.seed = seed
        self.steps_per_epoch = max(n_prompts // batch_size, 1)
        self._cached_epoch = None
        self._order = None

    def _order_for_epoch(self, epoch: int) -> np.ndarray:
        if epoch != self._cached_epoch:
            self._order = np.random.default_rng(self.seed + epoch).permutation(self.n)
            self._cached_epoch = epoch
        return self._order

    def get_batch(self, batch_index: int) -> list:
        epoch, within = divmod(batch_index, self.steps_per_epoch)
        start = within * self.batch_size
        return self._order_for_epoch(epoch)[start:start + self.batch_size].tolist()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f'not JSON serializable: {type(o)}')


def save_checkpoint(ckpt_dir: Path, model, optimizer, scheduler, gate, global_step: int) -> None:
    """Write to <ckpt>.tmp, then swap, so an interruption mid-save never leaves a broken checkpoint."""
    import torch
    tmp = ckpt_dir.with_name(ckpt_dir.name + '.tmp')
    old = ckpt_dir.with_name(ckpt_dir.name + '.old')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp / 'adapter'))
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'np_rng': np.random.get_state(),
        'py_rng': random.getstate(),
    }, tmp / 'trainer_state.pt')
    (tmp / 'run_state.json').write_text(json.dumps(
        {'global_step': global_step, 'gate_state': gate.state_dict()}, default=_json_default))
    if ckpt_dir.exists():
        if old.exists():
            shutil.rmtree(old)
        ckpt_dir.rename(old)
    tmp.rename(ckpt_dir)
    if old.exists():
        shutil.rmtree(old)


def find_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """The live checkpoint, or the previous one if a crash happened mid-swap."""
    for cand in (ckpt_dir, ckpt_dir.with_name(ckpt_dir.name + '.old')):
        if (cand / 'run_state.json').exists() and (cand / 'trainer_state.pt').exists():
            return cand
    return None


def load_checkpoint(ckpt: Path, model, optimizer, scheduler, gate) -> int:
    import torch
    from peft import load_peft_weights, set_peft_model_state_dict
    device = next(model.parameters()).device
    weights = load_peft_weights(str(ckpt / 'adapter'), device=str(device))
    set_peft_model_state_dict(model, weights)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    state = torch.load(ckpt / 'trainer_state.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    np.random.set_state(state['np_rng'])
    random.setstate(state['py_rng'])
    run_state = json.loads((ckpt / 'run_state.json').read_text())
    gate.load_state_dict(run_state['gate_state'])
    return int(run_state['global_step'])


class S3Syncer:
    """Fire-and-forget `aws s3 sync`; at most one sync in flight. No-op without a URI."""

    def __init__(self, s3_uri: Optional[str], log=print):
        self.s3_uri = s3_uri.rstrip('/') if s3_uri else None
        self.log = log
        self.proc = None
        if self.s3_uri and shutil.which('aws') is None:
            log('[s3] WARNING: aws CLI not found; S3 sync disabled.')
            self.s3_uri = None

    def sync(self, local_dir: Path, remote_suffix: str, wait: bool = False) -> None:
        if not self.s3_uri:
            return
        if self.proc is not None and self.proc.poll() is None:
            if not wait:
                return
            self.proc.wait()
        cmd = ['aws', 's3', 'sync', str(local_dir), f'{self.s3_uri}/{remote_suffix}', '--only-show-errors']
        self.proc = subprocess.Popen(cmd)
        if wait:
            rc = self.proc.wait()
            if rc != 0:
                self.log(f'[s3] WARNING: sync exited with code {rc}')
