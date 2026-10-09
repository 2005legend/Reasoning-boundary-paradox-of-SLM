"""Dataset loading and prompt formatting (Req 4)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .rewards import extract_last_boxed

# The standard Qwen math instruction (as in Yue et al. 2025 / ReCo), so results are comparable
# with the literature. No tag format is requested: the reward is correctness-only.
SYSTEM_PROMPT = 'Please reason step by step, and put your final answer within \\boxed{}.'

MATH_DATASET = 'EleutherAI/hendrycks_math'
MATH_SUBJECT_CONFIGS = ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
                        'number_theory', 'prealgebra', 'precalculus')


@dataclass
class Problem:
    prompt_id: str
    prompt_text: str
    chat_prompt: str
    ground_truth: str
    subject: str = ''
    level: int = 0


def _extract_gsm8k_answer(answer_field: str) -> str:
    match = re.search(r'####\s*([\-0-9,./]+)', answer_field)
    if not match:
        raise ValueError(f"Could not find '#### <answer>' in: {answer_field!r}")
    return match.group(1).replace(',', '').strip()


def format_chat_prompt(tokenizer, question: str) -> str:
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_math_level(level_field: str) -> int:
    """'Level 3' -> 3; unknown levels ('Level ?') -> 0."""
    m = re.search(r'(\d+)', str(level_field))
    return int(m.group(1)) if m else 0


def math_rows_to_problems(rows, tokenizer, split: str, levels=None) -> list:
    """rows: iterable of dicts with problem/solution/type/level. Skips rows without a boxed answer."""
    problems = []
    for row in rows:
        level = parse_math_level(row['level'])
        if levels and level not in levels:
            continue
        gt = extract_last_boxed(row['solution'])
        if not gt:
            continue
        problems.append(Problem(prompt_id=f'math_{split}_{len(problems)}', prompt_text=row['problem'],
                                chat_prompt=format_chat_prompt(tokenizer, row['problem']),
                                ground_truth=gt, subject=row['type'], level=level))
    return problems


def load_math_train(tokenizer, levels=None) -> list:
    """MATH train split (7,500 problems, disjoint from MATH-500, which is drawn from the test split)."""
    from datasets import load_dataset
    rows = []
    for cfg in MATH_SUBJECT_CONFIGS:
        rows.extend(load_dataset(MATH_DATASET, cfg, split='train'))
    return math_rows_to_problems(rows, tokenizer, 'train', levels)


def _gsm8k_like(tokenizer, dataset_name: str, config: str | None, split: str, prefix: str) -> list:
    from datasets import load_dataset
    ds = load_dataset(dataset_name, config, split=split) if config else load_dataset(dataset_name, split=split)
    return [Problem(prompt_id=f'{prefix}_{split}_{i}', prompt_text=row['question'],
                    chat_prompt=format_chat_prompt(tokenizer, row['question']),
                    ground_truth=_extract_gsm8k_answer(row['answer']))
            for i, row in enumerate(ds)]


def load_gsm8k(tokenizer, split: str = 'train') -> list:
    return _gsm8k_like(tokenizer, 'openai/gsm8k', 'main', split, 'gsm8k')


def load_gsm8k_platinum(tokenizer, split: str = 'test') -> list:
    return _gsm8k_like(tokenizer, 'madrylab/gsm8k-platinum', None, split, 'platinum')


def load_math500(tokenizer) -> list:
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    return [Problem(prompt_id=f'math500_{i}', prompt_text=row['problem'],
                    chat_prompt=format_chat_prompt(tokenizer, row['problem']),
                    ground_truth=str(row['answer']).strip(), subject=row.get('subject', ''),
                    level=int(row.get('level', 0) or 0))
            for i, row in enumerate(ds)]


def load_train_problems(name: str, tokenizer, levels=None) -> list:
    if name == 'math':
        return load_math_train(tokenizer, levels)
    if name == 'gsm8k':
        return load_gsm8k(tokenizer, split='train')
    raise ValueError(f'Unknown training dataset {name!r}')


def load_benchmark(name: str, tokenizer) -> list:
    if name == 'gsm8k':
        return load_gsm8k(tokenizer, split='test')
    if name == 'gsm8k_platinum':
        return load_gsm8k_platinum(tokenizer, split='test')
    if name == 'math500':
        return load_math500(tokenizer)
    raise ValueError(f'Unknown benchmark {name!r}')
