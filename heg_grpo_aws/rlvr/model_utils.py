"""bf16 LoRA model loading, rollout generation, and exact token-level log-prob scoring."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

QWEN_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation='sdpa')
    return model.to(device)


def load_lora_model(model_name: str, device: str, lora_r: int = 16, lora_alpha: int = 32,
                    lora_dropout: float = 0.0, gradient_checkpointing: bool = False):
    """bf16 base weights + fp32 LoRA adapters (bf16 cannot represent lr~1e-6 updates)."""
    import torch
    from peft import LoraConfig, get_peft_model
    model = load_base_model(model_name, device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          target_modules=QWEN_TARGET_MODULES, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def stop_token_ids(model, tokenizer) -> set:
    ids = set()
    eos = getattr(getattr(model, 'generation_config', None), 'eos_token_id', None)
    if isinstance(eos, int):
        ids.add(eos)
    elif eos:
        ids.update(eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def trim_completion(token_ids: list, stop_ids: set) -> tuple:
    """Cut at the first stop token (kept, so the model learns to stop). Returns (ids, truncated)."""
    for i, t in enumerate(token_ids):
        if t in stop_ids:
            return token_ids[:i + 1], False
    return token_ids, True


def report_vram(tag: str = '') -> str:
    import torch
    if not torch.cuda.is_available():
        return f'[{tag}] CUDA not available.'
    used = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return f'[{tag}] allocated={used:.2f}GB peak={peak:.2f}GB'


@dataclass
class RolloutBatch:
    prompt_ids: list           # record prompt_id, one per completion
    cluster_ids: list
    ground_truths: list
    prompt_token_ids: list     # list[list[int]], one per completion
    completion_token_ids: list # list[list[int]], trimmed at the first stop token
    completions: list          # decoded text
    truncated: list            # hit max_new_tokens without stopping
    group_size: int
    greedy_completions: list = None  # one per prompt when generated with greedy_extra=True


class GreedyRowsProcessor:
    """LogitsProcessor that pins chosen batch rows to their argmax token, so one sampling call can
    also produce greedy completions. Sampling transforms in use (temperature, top-p, top_k=0,
    repetition_penalty=1) all preserve the argmax, so wherever this runs in the processor chain the
    pinned rows decode exactly as do_sample=False would."""

    def __init__(self, greedy_rows):
        self.greedy_rows = greedy_rows  # bool tensor [batch]

    def __call__(self, input_ids, scores):
        import torch
        if not bool(self.greedy_rows.any()):
            return scores
        pinned = torch.full_like(scores, float('-inf'))
        top = scores.argmax(dim=-1, keepdim=True)
        pinned.scatter_(1, top, scores.gather(1, top))
        return torch.where(self.greedy_rows.unsqueeze(1), pinned, scores)


def generate_rollouts(model, tokenizer, prompt_records: list, group_size: int, max_new_tokens: int,
                      temperature: float, top_p: float, top_k: int = 0, repetition_penalty: float = 1.0,
                      microbatch_prompts: int = 8, greedy_extra: bool = False) -> RolloutBatch:
    """`group_size` samples per prompt, contiguous per prompt (the partitions rely on it).
    greedy_extra adds one greedy completion per prompt in the same generate call; it is returned in
    greedy_completions and is never part of the GRPO group."""
    import torch
    from transformers import LogitsProcessorList
    if greedy_extra and (repetition_penalty != 1.0 or top_k not in (0, None)):
        raise ValueError('greedy_extra needs repetition_penalty=1.0 and top_k=0 (argmax-preserving sampling).')
    was_training = model.training
    model.eval()
    pad_id = tokenizer.pad_token_id
    stops = stop_token_ids(model, tokenizer)
    device = next(model.parameters()).device
    rows_per_prompt = group_size + (1 if greedy_extra else 0)

    prompt_tok = tokenizer([r['chat_prompt'] for r in prompt_records], add_special_tokens=False).input_ids
    out = RolloutBatch([], [], [], [], [], [], [], group_size, [] if greedy_extra else None)
    with torch.no_grad():
        for start in range(0, len(prompt_records), microbatch_prompts):
            recs = prompt_records[start:start + microbatch_prompts]
            toks = prompt_tok[start:start + microbatch_prompts]
            width = max(len(t) for t in toks)
            rows = [[pad_id] * (width - len(t)) + t for t in toks for _ in range(rows_per_prompt)]
            masks = [[0] * (width - len(t)) + [1] * len(t) for t in toks for _ in range(rows_per_prompt)]
            input_ids = torch.tensor(rows, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            extra = {}
            if greedy_extra:
                greedy_rows = torch.tensor([g == group_size for _ in toks for g in range(rows_per_prompt)],
                                           device=device)
                extra['logits_processor'] = LogitsProcessorList([GreedyRowsProcessor(greedy_rows)])
            gen = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=temperature, top_p=top_p, top_k=top_k,
                                 repetition_penalty=repetition_penalty, pad_token_id=pad_id, **extra)
            new_tokens = gen[:, width:].tolist()
            for i, r in enumerate(recs):
                base = i * rows_per_prompt
                for g in range(group_size):
                    comp, trunc = trim_completion(new_tokens[base + g], stops)
                    out.prompt_ids.append(r['prompt_id'])
                    out.cluster_ids.append(r.get('cluster_id', 0))
                    out.ground_truths.append(r['ground_truth'])
                    out.prompt_token_ids.append(toks[i])
                    out.completion_token_ids.append(comp)
                    out.completions.append(tokenizer.decode(comp, skip_special_tokens=True))
                    out.truncated.append(trunc)
                if greedy_extra:
                    comp, _ = trim_completion(new_tokens[base + group_size], stops)
                    out.greedy_completions.append(tokenizer.decode(comp, skip_special_tokens=True))
            del gen, input_ids, attention_mask
    if was_training:
        model.train()
    return out


def build_scoring_batch(prompt_ids_list: list, completion_ids_list: list, pad_id: int, device):
    """Right-padded [prompt + completion] rows, cropped to the region that predicts completion tokens.

    Returns input_ids [B, L], attention_mask [B, L], logits_to_keep N, target_ids [B, N-1],
    loss_mask [B, N-1]. Logit position j predicts token j+1, so only the last N positions
    (starting one before the shortest prompt ends) need lm_head; vocab is 152k, so this matters.
    """
    import torch
    seqs = [p + c for p, c in zip(prompt_ids_list, completion_ids_list)]
    B, L = len(seqs), max(len(s) for s in seqs)
    min_prompt = min(len(p) for p in prompt_ids_list)
    first_pred = min_prompt - 1
    n_keep = L - first_pred
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    loss_mask = torch.zeros((B, n_keep - 1), dtype=torch.float32)
    for r, (p, c) in enumerate(zip(prompt_ids_list, completion_ids_list)):
        s = p + c
        input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
        attention_mask[r, :len(s)] = 1
        # completion token at absolute position t is predicted by logit t-1 -> local index t-1-first_pred
        lo = len(p) - 1 - first_pred
        loss_mask[r, lo:lo + len(c)] = 1.0
    target_ids = input_ids[:, first_pred + 1:]
    return (input_ids.to(device), attention_mask.to(device), n_keep,
            target_ids.to(device), loss_mask.to(device))


def token_logprobs(model, input_ids, attention_mask, n_keep: int, target_ids):
    """log pi(target) for the kept region, computed in fp32. Shape [B, n_keep-1]."""
    import torch
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                logits_to_keep=n_keep)
    logits = out.logits[:, :-1, :].float()
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            target_ids.reshape(-1), reduction='none')
    return -nll.view(target_ids.shape)


def score_completions(model, tokenizer, prompt_ids_list: list, completion_ids_list: list,
                      chunk_size: int = 8, disable_adapter: bool = False) -> list:
    """No-grad mean per-token log-prob of each completion (probing / diagnostics, Req 40)."""
    import contextlib
    import torch
    device = next(model.parameters()).device
    ctx = model.disable_adapter() if disable_adapter else contextlib.nullcontext()
    means = []
    was_training = model.training
    model.eval()
    with torch.no_grad(), ctx:
        for s in range(0, len(prompt_ids_list), chunk_size):
            ids, attn, n_keep, tgt, mask = build_scoring_batch(
                prompt_ids_list[s:s + chunk_size], completion_ids_list[s:s + chunk_size],
                tokenizer.pad_token_id, device)
            lp = token_logprobs(model, ids, attn, n_keep, tgt)
            means.extend(((lp * mask).sum(1) / mask.sum(1).clamp(min=1.0)).tolist())
    if was_training:
        model.train()
    return means


@contextmanager
def generation_copy(model, enabled: bool = True):
    """Yield a plain copy of a LoRA model with the current adapters merged in, for sampling.

    Measured on the A10G at 144 rows: 61.5 ms per decode step vs 78.9 ms through the LoRA layers.
    The training model is never modified: the copy is rebuilt from the current weights each call, so
    bf16 merge rounding cannot accumulate in the base weights (as repeated merge/unmerge would)."""
    import copy
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = ()
    if not enabled or not isinstance(model, PeftModel):
        yield model
        return
    gen = copy.deepcopy(model).merge_and_unload()
    gen.eval()
    try:
        yield gen
    finally:
        del gen


def max_relative_logit_gap(model_a, model_b, input_ids) -> float:
    """max |logits_a - logits_b| / max |logits_a| on one input; used to verify the merged copy."""
    import torch
    with torch.no_grad():
        la = model_a(input_ids=input_ids).logits.float()
        lb = model_b(input_ids=input_ids).logits.float()
    return float((la - lb).abs().max() / (la.abs().max() + 1e-6))


def load_adapter_for_inference(model_name: str, adapter_dir: Optional[str], device: str):
    """Base model, or base + LoRA merged into the weights (faster generation for eval)."""
    model = load_base_model(model_name, device)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.eval()
    return model
