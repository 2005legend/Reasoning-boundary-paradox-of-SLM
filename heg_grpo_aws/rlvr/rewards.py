"""Output parsing and rewards (Req 11-13, 36-38). CPU only; never raises."""
from __future__ import annotations

import logging
import re
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application, parse_expr, standard_transformations,
)

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    # Its own timeouts are disabled below (we apply one SIGALRM limit around both calls),
    # which makes it warn on every call.
    logging.getLogger('math_verify').setLevel(logging.ERROR)
except ImportError:
    _mv_parse = _mv_verify = None

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SYMPY_TIMEOUT_S = 2.0
_MAX_EXPR_LEN = 80
_SAFE_CHARS = re.compile(r'^[0-9A-Za-z+\-*/().\s]*$')
_ALLOWED_WORDS = {'sqrt', 'pi', 'e', 'i', 'x', 'y', 'z', 'a', 'b', 'c', 'n', 'k', 'm', 't'}


@dataclass
class ParsedOutput:
    reasoning: Optional[str]
    answer: Optional[str]
    boxed: Optional[str]

    @property
    def is_well_formed(self) -> bool:
        return self.reasoning is not None and self.answer is not None and self.boxed is not None


def _extract_outermost(text: str, opening: str, closing: str) -> Optional[str]:
    """Req 36.5: for nested/duplicate tags, take the outermost span."""
    start = text.find(opening)
    if start == -1:
        return None
    end = text.rfind(closing)
    if end == -1 or end <= start:
        return None
    return text[start + len(opening):end].strip()


def extract_last_boxed(text: Optional[str]) -> Optional[str]:
    """Content of the last balanced \\boxed{...}; handles nested braces like \\boxed{\\frac{1}{2}}."""
    if not text:
        return None
    idx = text.rfind('\\boxed')
    while idx != -1:
        j = idx + len('\\boxed')
        while j < len(text) and text[j] == ' ':
            j += 1
        if j < len(text) and text[j] == '{':
            depth = 0
            for k in range(j, len(text)):
                if text[k] == '{':
                    depth += 1
                elif text[k] == '}':
                    depth -= 1
                    if depth == 0:
                        return text[j + 1:k].strip()
        idx = text.rfind('\\boxed', 0, idx)
    return None


def parse_output(text: Optional[str]) -> ParsedOutput:
    """Req 36. Falls back to scanning the raw text for \\boxed{} when the model skips the
    <answer> tag (small instruct models do this by default); format_reward stays strict."""
    if text is None:
        return ParsedOutput(None, None, None)
    reasoning = _extract_outermost(text, '<reasoning>', '</reasoning>')
    answer_block = _extract_outermost(text, '<answer>', '</answer>')
    boxed = extract_last_boxed(answer_block) if answer_block is not None else None
    if boxed is None:
        boxed = extract_last_boxed(text)
    return ParsedOutput(reasoning=reasoning, answer=answer_block, boxed=boxed)


def pretty_print(reasoning: str, answer: str, boxed_value: str) -> str:
    """Req 37: inverse of parse_output."""
    reasoning = (reasoning or '').strip()
    boxed_value = (boxed_value or '').strip()
    answer_body = (answer or '').strip()
    if f'\\boxed{{{boxed_value}}}' not in answer_body:
        answer_body = f'{answer_body}\n\\boxed{{{boxed_value}}}'.strip()
    return f'<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer_body}\n</answer>'


def format_reward(generated_text: str) -> float:
    """Req 11: 1.0 iff reasoning + answer + boxed are all present."""
    return 1.0 if parse_output(generated_text).is_well_formed else 0.0


def _clean_latex(s: str) -> str:
    s = s.strip().strip('$').strip()
    no_text = re.sub(r'\\text\{[^{}]*\}', '', s)
    s = no_text if no_text.strip() else re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
    for tok in ('\\left', '\\right', '\\!', '\\,', '\\;', '\\ ', '^\\circ', '^{\\circ}', '\\circ',
                '\\%', '%', '\\$'):
        s = s.replace(tok, '')
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
        s = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', s)
    s = s.replace('\\cdot', '*').replace('\\times', '*').replace('\\pi', 'pi')
    s = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', s)  # thousands separators only
    s = s.replace('{', '(').replace('}', ')').replace('^', '**')
    return s.rstrip('.').strip()


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


def _is_safe_for_sympy(s: str) -> bool:
    """parse_expr eval()s its input, and this input is model-generated: whitelist it."""
    if not s or len(s) > _MAX_EXPR_LEN or '__' in s or not _SAFE_CHARS.match(s):
        return False
    if s.count('**') > 2 or re.search(r'\*\*\s*\(?\s*\d{3,}', s):
        return False
    return all(w in _ALLOWED_WORDS for w in re.findall(r'[A-Za-z]+', s))


@contextmanager
def _time_limit(seconds: float):
    """SIGALRM-based limit (Linux main thread); no-op elsewhere."""
    usable = hasattr(signal, 'setitimer') and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return

    def _raise(signum, frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _math_verify_equivalent(predicted: str, ground_truth: str) -> Optional[bool]:
    """Math-Verify (LaTeX-aware, no eval of the input); None if unavailable or it fails."""
    if _mv_parse is None:
        return None
    try:
        with _time_limit(_SYMPY_TIMEOUT_S):
            gold = _mv_parse('$' + ground_truth + '$', parsing_timeout=None)
            pred = _mv_parse('$' + predicted + '$', parsing_timeout=None)
            if not gold or not pred:
                return None
            return bool(_mv_verify(gold, pred, timeout_seconds=None))
    except Exception:
        return None


# The symbolic path (Math-Verify, sympy) runs in a separate worker process with a memory cap and a
# wall-clock limit. A model answer such as 2^{10^{9}} makes sympy build an enormous integer inside
# one C call, which SIGALRM cannot interrupt; in the main process that ends with the kernel's OOM
# killer taking down the whole training run (seen twice at seed 1, step ~85, 2026-10-06). In the
# worker it raises MemoryError or times out, the answer counts as wrong, and the worker is replaced.
_GUARD_MEM_BYTES = 3 << 30
_GUARD_TIMEOUT_S = 10.0
_guard_pool = None
_guard_lock = threading.Lock()


def _guard_init():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (_GUARD_MEM_BYTES, _GUARD_MEM_BYTES))
    except (ImportError, ValueError, OSError):
        pass  # no RLIMIT on this platform: the wall-clock limit still applies


def _guard_reset():
    global _guard_pool
    pool, _guard_pool = _guard_pool, None
    if pool is not None:
        for proc in list(getattr(pool, '_processes', {}).values()):
            proc.kill()
        pool.shutdown(wait=False, cancel_futures=True)


def _guarded_symbolic_equivalent(predicted: str, ground_truth: str) -> bool:
    global _guard_pool
    import concurrent.futures as cf
    import multiprocessing as mp
    with _guard_lock:
        if _guard_pool is None:
            _guard_pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'),
                                                 initializer=_guard_init)
        try:
            return bool(_guard_pool.submit(_symbolic_equivalent, predicted, ground_truth)
                        .result(timeout=_GUARD_TIMEOUT_S))
        except Exception:  # timeout, MemoryError, or the worker died
            _guard_reset()
            return False


def answers_equivalent(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """Req 12.4: exact/numeric fast path -> Math-Verify -> guarded sympy. False on any failure."""
    if predicted is None or ground_truth is None:
        return False
    p, g = _clean_latex(predicted), _clean_latex(ground_truth)
    if p.replace(' ', '') == g.replace(' ', ''):
        return True
    fp, fg = _to_float(p), _to_float(g)
    if fp is not None and fg is not None:
        return abs(fp - fg) <= 1e-6 * max(1.0, abs(fg))
    return _guarded_symbolic_equivalent(predicted, ground_truth)


def _symbolic_equivalent(predicted: str, ground_truth: str) -> bool:
    """Math-Verify, then guarded sympy. Runs inside the guard worker."""
    p, g = _clean_latex(predicted), _clean_latex(ground_truth)
    mv = _math_verify_equivalent(predicted, ground_truth)
    if mv is not None:
        return mv
    if not (_is_safe_for_sympy(p) and _is_safe_for_sympy(g)):
        return False
    try:
        with _time_limit(_SYMPY_TIMEOUT_S):
            pe = parse_expr(p, transformations=_TRANSFORMS, evaluate=True)
            ge = parse_expr(g, transformations=_TRANSFORMS, evaluate=True)
            if sympy.simplify(pe - ge) == 0:
                return True
            return bool(abs(complex((pe - ge).evalf())) < 1e-6)
    except Exception:
        return False


def correctness_reward(is_correct: bool, reward_mode: str = 'positive_only') -> float:
    if reward_mode == 'positive_only':
        return 1.0 if is_correct else 0.0
    if reward_mode == 'negative_only':
        return 0.0 if is_correct else -1.0
    if reward_mode == 'hybrid':
        return 1.0 if is_correct else -0.5
    raise ValueError(f'Unknown reward_mode: {reward_mode}')


@dataclass
class RewardBreakdown:
    format: float
    correctness: float
    total: float
    is_correct: bool
    is_well_formed: bool


def compute_reward(generated_text: str, ground_truth: str, reward_mode: str = 'positive_only',
                   format_weight: float = 1.0, correctness_weight: float = 1.0) -> RewardBreakdown:
    """Req 13: total = format_weight * format + correctness_weight * correctness."""
    parsed = parse_output(generated_text)
    is_correct = answers_equivalent(parsed.boxed, ground_truth)
    f = 1.0 if parsed.is_well_formed else 0.0
    c = correctness_reward(is_correct, reward_mode)
    return RewardBreakdown(format=f, correctness=c, total=format_weight * f + correctness_weight * c,
                           is_correct=is_correct, is_well_formed=parsed.is_well_formed)
