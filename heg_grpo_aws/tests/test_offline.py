"""CPU-only tests. Run before any paid GPU time: python -m pytest -q tests"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvr.config import GATE_TYPES, ExperimentConfig, GateConfig
from rlvr.data import math_rows_to_problems, parse_math_level
from rlvr.evaluation import (bootstrap_delta_slope_ci, compute_pass_at_k, compute_shrinkage_slope,
                             gini_coefficient, pass_at_k_unbiased)
from rlvr.gates import (BBGGate, CBGRPOGate, CompositeGate, GateContext, MEGGate, SELFGate, VanillaGate,
                        bbg_utility, build_gate, compute_group_partitions, randomize_partition)
from rlvr.model_utils import build_scoring_batch, trim_completion
from rlvr.rewards import _mv_parse, answers_equivalent, compute_reward, extract_last_boxed, parse_output
from rlvr.trainer_utils import PromptSampler


def ctx(cluster_ids, is_correct, group_size, partition=None, greedy_correct=None):
    n = len(cluster_ids)
    return GateContext.build(cluster_ids, list(range(n)), is_correct, group_size, partition, greedy_correct)


# ── config ──────────────────────────────────────────────────────────────────

def test_config_rules():
    assert ExperimentConfig(model_size='1.5B', gate=GateConfig(gate_type='h_cb_grpo')).training_steps == 800
    for gate in GATE_TYPES:  # the 0.5B restriction on the SELF family is lifted (decision of 2026-10-03)
        ExperimentConfig(model_size='0.5B', gate=GateConfig(gate_type=gate))
    cfg = ExperimentConfig(model_size='0.5B', gate=GateConfig(gate_type='heg_grpo'), steps_override=123)
    assert cfg.training_steps == 123 and cfg.model_name == 'Qwen/Qwen2.5-0.5B-Instruct'
    assert cfg.train_dataset == 'math' and cfg.in_domain_benchmark == 'math500'
    assert cfg.reward.format_weight == 0.0
    with pytest.raises(ValueError):
        ExperimentConfig(train_dataset='gsm8k')  # subject topics need MATH
    with pytest.raises(ValueError):
        GateConfig(gate_type='meg', meg_partition='random').validate('0.5B')


def test_config_json_roundtrip(tmp_path):
    cfg = ExperimentConfig(model_size='0.5B', gate=GateConfig(gate_type='meg', meg_random_partition=True),
                           tag='x', seed=2)
    cfg.to_json(str(tmp_path / 'c.json'))
    back = ExperimentConfig.from_json(str(tmp_path / 'c.json'))
    assert back.run_name == 'meg__x' and back.seed == 2 and back.gate.meg_random_partition
    assert back.math_levels == [1, 2, 3, 4, 5]


# ── rewards ─────────────────────────────────────────────────────────────────

def test_boxed_nested_and_last():
    assert extract_last_boxed(r'\boxed{\frac{1}{2}}') == r'\frac{1}{2}'
    assert extract_last_boxed(r'first \boxed{3} then final \boxed{42}') == '42'
    assert extract_last_boxed(r'unbalanced \boxed{3') is None
    assert extract_last_boxed('no box here') is None


def test_parse_output_and_correctness_only_reward():
    bare = r'Some reasoning. The answer is \boxed{18}.'
    parsed = parse_output(bare)
    assert parsed.boxed == '18' and not parsed.is_well_formed
    r = compute_reward(bare, '18', format_weight=0.0)
    assert r.is_correct and r.total == 1.0 and r.format == 0.0
    assert compute_reward(r'\boxed{17}', '18', format_weight=0.0).total == 0.0


@pytest.mark.parametrize('pred,gt,ok', [
    ('18', '18', True), ('18.0', '18', True), ('1,000', '1000', True), ('$18', '18', True),
    ('50\\%', '50', True), ('17', '18', False), (r'\frac{1}{2}', '0.5', True),
    (r'\dfrac{14}{3}', r'\frac{14}{3}', True), (r'\sqrt{4}', '2', True), ('2\\pi', '2pi', True),
    (r'5 \text{ cm}', '5', True), ('(3, 4)', '(3,4)', True), (None, '1', False),
])
def test_answers_equivalent(pred, gt, ok):
    assert answers_equivalent(pred, gt) is ok


def test_math_verify_installed():
    # Without it MATH rewards silently fall back to the weaker sympy check.
    assert _mv_parse is not None, 'math-verify is missing: uv pip install -r requirements.txt'


@pytest.mark.parametrize('pred,gt,ok', [
    (r'(-\infty,3]', r'(-\infty, 3]', True), (r'\sqrt{8}', r'2\sqrt{2}', True), ('30', r'30^\circ', True),
    ('B', r'\text{(B)}', True), (r'\begin{pmatrix}1\\2\end{pmatrix}', r'\begin{pmatrix} 1 \\ 2 \end{pmatrix}', True),
    (r'1+x^2', r'x^2+1', True), (r'\frac{\pi}{2}', r'\pi/2', True), ('4', '3', False),
    (r'(-\infty,2]', r'(-\infty, 3]', False),
])
def test_math_answers_via_math_verify(pred, gt, ok):
    assert answers_equivalent(pred, gt) is ok


def test_unsafe_and_explosive_input_never_hangs_or_matches():
    assert answers_equivalent("__import__('os').system('echo hi')", '1') is False
    assert answers_equivalent('9^9^9^9', '1') is False
    assert answers_equivalent('2^100000', '1') is False


def test_memory_bomb_answers_are_contained():
    # LaTeX towers that make sympy build gigantic integers inside one C call (SIGALRM cannot stop
    # that). They must come back False within the guard's time limit, and the checker must still
    # work afterwards (the worker is replaced).
    import time
    t0 = time.time()
    for bomb in ('2^{10^{10}}', '10^{10^{9}}', '(10^{10^{8}})!', '9^{9^{9^{9}}}'):
        assert answers_equivalent(bomb, '5') is False
    assert time.time() - t0 < 60
    assert answers_equivalent(r'\frac{\sqrt{2}}{2}', r'\frac{1}{\sqrt{2}}') is True


# ── MATH data ───────────────────────────────────────────────────────────────

class _FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return '|'.join(m['content'] for m in messages)


def test_math_rows_to_problems():
    rows = [
        {'problem': 'p1', 'solution': r'so \boxed{\frac{1}{2}}', 'type': 'Algebra', 'level': 'Level 2'},
        {'problem': 'p2', 'solution': 'no boxed answer', 'type': 'Geometry', 'level': 'Level 1'},
        {'problem': 'p3', 'solution': r'\boxed{3} then \boxed{5}', 'type': 'Precalculus', 'level': 'Level 5'},
        {'problem': 'p4', 'solution': r'\boxed{7}', 'type': 'Algebra', 'level': 'Level ?'},
    ]
    probs = math_rows_to_problems(rows, _FakeTok(), 'train')
    assert [p.ground_truth for p in probs] == [r'\frac{1}{2}', '5', '7']
    assert [p.subject for p in probs] == ['Algebra', 'Precalculus', 'Algebra']
    assert [p.level for p in probs] == [2, 5, 0]
    assert len({p.prompt_id for p in probs}) == 3 and probs[0].chat_prompt.endswith('|p1')
    assert [p.prompt_text for p in math_rows_to_problems(rows, _FakeTok(), 'train', levels=[1, 2, 3])] == ['p1']
    assert parse_math_level('Level 4') == 4 and parse_math_level('Level ?') == 0


# ── evaluation ──────────────────────────────────────────────────────────────

def test_pass_at_k():
    assert pass_at_k_unbiased(10, 0, 1) == 0.0
    assert pass_at_k_unbiased(10, 10, 5) == 1.0
    assert abs(pass_at_k_unbiased(10, 1, 1) - 0.1) < 1e-12
    pk = compute_pass_at_k(np.array([[1, 0, 0, 0], [0, 0, 0, 0]]), [1, 4])
    assert abs(pk[1] - 0.125) < 1e-12 and pk[4] == 0.5


def test_shrinkage_slope_sign_and_string_keys():
    base = {'1': 0.2, '2': 0.3, '4': 0.4, '8': 0.5}
    rl = {1: 0.4, 2: 0.4, 4: 0.4, 8: 0.4}
    assert compute_shrinkage_slope(base, rl).slope < 0


def test_analysis_helpers_match_reference_implementations():
    from analyze import per_problem_pass_at_k, slopes_from_rows
    rng = np.random.default_rng(0)
    ks = [1, 2, 4, 8]
    base_m = (rng.random((40, 8)) < rng.random((40, 1))).astype(int)
    run_m = (rng.random((40, 8)) < rng.random((40, 1))).astype(int)
    base_rows, run_rows = per_problem_pass_at_k(base_m, ks), per_problem_pass_at_k(run_m, ks)
    assert np.allclose(base_rows.mean(0), list(compute_pass_at_k(base_m, ks).values()))
    ref = compute_shrinkage_slope(compute_pass_at_k(base_m, ks), compute_pass_at_k(run_m, ks)).slope
    assert abs(slopes_from_rows(run_rows, base_rows, np.log(np.array(ks, float))) - ref) < 1e-9


def test_bootstrap_ci():
    res = bootstrap_delta_slope_ci([0.1, 0.11, 0.12], [-0.1, -0.11, -0.09])
    assert res.excludes_zero and res.mean_delta > 0
    with pytest.raises(ValueError):
        bootstrap_delta_slope_ci([0.1], [0.0, 0.1])
    assert gini_coefficient(np.ones(5)) == 0.0


# ── gates ───────────────────────────────────────────────────────────────────

def test_vanilla_gate():
    assert np.allclose(VanillaGate().step(ctx([0, 1, 2], [1, 0, 1], 3), [1.0, -1.0, 2.0]), 1.0)


def test_cbgrpo_throttles_and_positive_mass_only():
    g = CBGRPOGate(n_clusters=4, theta=1.5, decay=0.9, ema_alpha=0.5)
    c = ctx([0, 0, 1, 1], [1, 1, 0, 0], 2)
    ws = [g.step(c, [2.0, 2.0, -0.5, -0.5])[0] for _ in range(30)]
    assert ws[0] == 1.0 and ws[-1] < ws[5] and g.spend_ema[1] == 0.0


def test_cbgrpo_mean_preserving_budget():
    cfg = GateConfig(gate_type='cb_grpo')
    g = build_gate('cb_grpo', 2, cfg)
    assert g.mean_preserving and abs(cfg.cb_decay ** (1.5 - cfg.cb_theta) - 0.5) < 0.01  # 1.5x mean -> half
    g.spend_ema = np.array([3.0, 1.0])  # topic 0 at 1.5x the mean, topic 1 at 0.5x
    # groups of 2: [correct, wrong] in topic 0, [correct, wrong] in topic 1, [correct, correct] in topic 1
    w = g.compute_weights(ctx([0, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 1], 2))
    pos = [0, 2]
    assert abs(w[pos].mean() - 1.0) < 1e-9                   # positive credit is moved, not removed
    assert w[0] < 1.0 < w[2] and abs(w[0] / w[2] - 0.5) < 0.01
    assert np.allclose(w[[1, 3, 4, 5]], 1.0)                 # negatives and zero-advantage groups untouched
    g.spend_ema = np.array([1.0, 1.0])
    assert np.allclose(g.compute_weights(ctx([0, 0, 1, 1], [1, 0, 1, 0], 2)), 1.0)  # in budget -> no-op


def test_self_gate_is_the_greedy_failure_filter():
    g = SELFGate(self_lambda=0.0)
    w = g.compute_weights(ctx([0] * 4, [1, 0, 1, 1], 2, greedy_correct=[True, True, False, False]))
    assert list(w) == [0.0, 0.0, 1.0, 1.0]
    assert np.allclose(g.compute_weights(ctx([0] * 2, [1, 0], 2)), 1.0)  # no greedy info -> no-op
    assert g.requires_greedy and not g.state_dict()


def test_meg_credit_redistribution():
    meg = MEGGate(alpha=0.8, clip_min=0.3, clip_max=3.0)
    # group of 6: correct rollouts 0-3 share mode 0 except rollout 3 (rare mode 1); 4,5 incorrect
    part = np.array([0, 0, 0, 1, -1, -1])
    w = meg.compute_weights(ctx([0] * 6, [1, 1, 1, 1, 0, 0], 6, partition=part))
    assert abs(w[:4].mean() - 1.0) < 1e-9          # total credit unchanged
    assert w[3] > 1.0 > w[0] and w[0] == w[1] == w[2]  # rare correct mode gets more
    assert np.allclose(w[4:], 1.0)                 # incorrect untouched
    one_mode = meg.compute_weights(ctx([0] * 4, [1, 1, 1, 0], 4, partition=np.array([0, 0, 0, -1])))
    assert np.allclose(one_mode, 1.0)              # nothing to redistribute
    assert np.allclose(meg.compute_weights(ctx([0] * 2, [1, 1], 2)), 1.0)
    alpha0 = MEGGate(alpha=0.0).compute_weights(ctx([0] * 6, [1, 1, 1, 1, 0, 0], 6, partition=part))
    assert np.allclose(alpha0, 1.0)                # alpha = 0 recovers GRPO


def test_bbg_utility_matches_paper_table_and_gate():
    assert abs(bbg_utility(0, 8, 16) - 6.0) < 1e-9  # arXiv:2606.15455 Table 4
    assert abs(bbg_utility(1, 8, 16) - 2.087) < 1e-3
    table = BBGGate(k_ref=16, tau=0.01).bucket_weights(8)
    assert table[1] == 1.0 and table[2] < table[1] and table[3] < table[2]
    assert np.all(table[4:] == 0.0)                 # hard-gated buckets (u < tau (n+1))
    w = BBGGate().compute_weights(ctx([0] * 16, [1] + [0] * 7 + [1] * 4 + [0] * 4, 8))
    assert np.allclose(w[:8], 1.0) and np.allclose(w[8:], 0.0)


def test_composites_and_factory():
    heg = build_gate('heg_grpo', 3, GateConfig(gate_type='heg_grpo'))
    hcb = build_gate('h_cb_grpo', 3, GateConfig(gate_type='h_cb_grpo'))
    assert heg.name == 'heg_grpo' and heg.requires_partition and not heg.requires_greedy
    assert hcb.name == 'h_cb_grpo' and hcb.requires_greedy and not hcb.requires_partition
    for gt in GATE_TYPES:
        assert build_gate(gt, 3, GateConfig(gate_type=gt)).name == gt
    c = ctx([0, 0, 1, 1], [1, 1, 1, 0], 4, partition=np.array([0, 1, 1, -1]))
    for _ in range(5):
        w = heg.step(c, [1.0, 1.0, 1.0, -3.0])
    assert w[0] > w[1] and w[3] == 1.0
    sd = json.loads(json.dumps(heg.state_dict(), default=lambda o: o.tolist()))
    heg2 = build_gate('heg_grpo', 3, GateConfig(gate_type='heg_grpo'))
    heg2.load_state_dict(sd)
    assert np.allclose(heg2.macro.spend_ema, heg.macro.spend_ema)


# ── solution-mode partitions ────────────────────────────────────────────────

def test_bigram_partition_and_diagnostics():
    a = 'We factor the quadratic as (x-2)(x-3) so the roots are 2 and 3 and the answer is 5.'
    b = 'We factor the quadratic as (x-2)(x-3) so the roots are 2 and 3, hence the answer is 5.'
    c = 'Using the formula for the sum of roots, -b/a gives 5 directly.'
    texts = [a, b, c, 'wrong', a, 'x', 'y', 'z']
    labels, diag = compute_group_partitions(texts, [1, 1, 1, 0, 1, 0, 0, 0], 4, method='bigram',
                                            bigram_threshold=0.5)
    assert labels[0] == labels[1] != labels[2] and labels[3] == -1
    assert labels[4] == 0 and list(labels[5:]) == [-1, -1, -1]   # singleton correct group
    assert diag['mean_correct_modes'] == 2.0 and 0 < diag['mean_group_mode_entropy'] < 1
    with pytest.raises(ValueError):
        compute_group_partitions(texts, [1] * 8, 4, method='random')


def test_randomize_partition_keeps_structure():
    labels = np.array([0, 0, 1, -1, 0, 0, 0, 0])
    out = randomize_partition(labels, 4, np.random.default_rng(0))
    assert out[3] == -1 and set(out[:3]) <= {0, 1} and set(out[4:]) == {0}


# ── sampler / token plumbing ────────────────────────────────────────────────

def test_sampler_deterministic_and_disjoint_within_epoch():
    s1, s2 = PromptSampler(100, 8, seed=3), PromptSampler(100, 8, seed=3)
    assert [s1.get_batch(i) for i in range(30)] == [s2.get_batch(i) for i in range(30)]
    epoch0 = [x for i in range(s1.steps_per_epoch) for x in s1.get_batch(i)]
    assert len(set(epoch0)) == len(epoch0)


def test_trim_completion():
    assert trim_completion([5, 6, 99, 0, 0], {99}) == ([5, 6, 99], False)
    assert trim_completion([5, 6, 7], {99}) == ([5, 6, 7], True)


def test_scoring_mask_selects_exactly_completion_tokens():
    pytest.importorskip('torch')
    prompts, comps = [[1, 2, 3], [4, 5, 6, 7, 8]], [[10, 11], [12, 13, 14]]
    ids, attn, n_keep, tgt, mask = build_scoring_batch(prompts, comps, pad_id=0, device='cpu')
    assert [tgt[r][mask[r].bool()].tolist() for r in range(2)] == comps
    assert attn.sum().item() == sum(len(p) + len(c) for p, c in zip(prompts, comps))


# ── model-level checks on a tiny random Qwen2 (no download) ─────────────────

def _tiny_model(lora=True):
    torch = pytest.importorskip('torch')
    peft = pytest.importorskip('peft')
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    base = Qwen2ForCausalLM(cfg)
    if not lora:
        return base
    lcfg = peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=['q_proj', 'v_proj'],
                           task_type='CAUSAL_LM', init_lora_weights=False)
    return peft.get_peft_model(base, lcfg)


def test_greedy_rows_decode_exactly_like_do_sample_false():
    torch = pytest.importorskip('torch')
    from transformers import LogitsProcessorList
    from rlvr.model_utils import GreedyRowsProcessor
    model = _tiny_model(lora=False).eval()
    prompt = torch.tensor([[5, 9, 13, 2, 7]])
    common = dict(max_new_tokens=12, pad_token_id=0, eos_token_id=63)
    ref = model.generate(prompt, do_sample=False, **common)[0, 5:].tolist()
    torch.manual_seed(1)
    out = model.generate(prompt.repeat(3, 1), do_sample=True, temperature=0.7, top_p=0.95, top_k=0,
                         repetition_penalty=1.0, logits_processor=LogitsProcessorList(
                             [GreedyRowsProcessor(torch.tensor([False, False, True]))]), **common)
    assert out[2, 5:].tolist() == ref
    assert out[0, 5:].tolist() != ref or out[1, 5:].tolist() != ref  # the sampled rows really sample


def test_token_logprobs_match_naive_full_sequence():
    torch = pytest.importorskip('torch')
    from rlvr.model_utils import token_logprobs
    model = _tiny_model().eval()
    prompts, comps = [[1, 2, 3, 4], [5, 6]], [[7, 8, 9], [10, 11, 12, 13]]
    ids, attn, n_keep, tgt, mask = build_scoring_batch(prompts, comps, pad_id=0, device='cpu')
    with torch.no_grad():
        lp = token_logprobs(model, ids, attn, n_keep, tgt)
        for r, (p, c) in enumerate(zip(prompts, comps)):
            full = torch.log_softmax(model(input_ids=torch.tensor([p + c])).logits[0, :-1].float(), -1)
            naive = full[torch.arange(len(p) - 1, len(p) + len(c) - 1), torch.tensor(c)]
            assert torch.allclose(lp[r][mask[r].bool()], naive, atol=1e-5)


def test_chunked_backward_equals_full_batch_gradient():
    torch = pytest.importorskip('torch')
    from rlvr.grpo_core import ppo_clip_token_loss
    from rlvr.model_utils import token_logprobs
    prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9], [3, 3, 3]]
    comps = [[10, 11], [12, 13, 14], [15], [16, 17, 18, 19]]
    adv = torch.tensor([1.0, -0.5, 0.3, -1.2])
    total = float(sum(len(c) for c in comps))

    def grads(chunk):
        model = _tiny_model().train()
        for s in range(0, 4, chunk):
            ids, attn, n_keep, tgt, mask = build_scoring_batch(prompts[s:s + chunk], comps[s:s + chunk], 0, 'cpu')
            lp = token_logprobs(model, ids, attn, n_keep, tgt)
            ((ppo_clip_token_loss(lp, lp.detach(), adv[s:s + chunk], 0.2) * mask).sum() / total).backward()
        return [p.grad.clone() for p in model.parameters() if p.requires_grad]

    for a, b in zip(grads(4), grads(1)):
        assert torch.allclose(a, b, atol=1e-6)


def test_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip('torch')
    from rlvr.trainer_utils import find_checkpoint, load_checkpoint, save_checkpoint
    model = _tiny_model()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s))
    gate = CompositeGate(CBGRPOGate(2), MEGGate())
    gate.macro.spend_ema[:] = [0.3, 0.7]
    sum(p.sum() for p in params).backward()
    opt.step(); sched.step()
    ckpt = tmp_path / 'checkpoint'
    save_checkpoint(ckpt, model, opt, sched, gate, global_step=7)
    save_checkpoint(ckpt, model, opt, sched, gate, global_step=7)  # overwrite path exercises the swap
    snapshot = [p.detach().clone() for p in params]

    model2 = _tiny_model()
    params2 = [p for p in model2.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(params2, lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0 / (1 + s))
    gate2 = CompositeGate(CBGRPOGate(2), MEGGate())
    assert load_checkpoint(find_checkpoint(ckpt), model2, opt2, sched2, gate2) == 7
    assert all(torch.allclose(a, b) for a, b in zip(snapshot, params2))
    assert np.allclose(gate2.macro.spend_ema, [0.3, 0.7])
    assert sched2.last_epoch == sched.last_epoch


# ── human evaluation tooling (scripts/human_eval.py) ─────────────────────────

def _human_eval_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('human_eval', Path(__file__).resolve().parents[1] / 'scripts' / 'human_eval.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_human_eval_statistics_match_references():
    he = _human_eval_module()
    scipy_stats = pytest.importorskip('scipy.stats')
    sk = pytest.importorskip('sklearn.metrics')
    a = ['S', 'S', 'D', 'D', 'S', 'D', 'S', 'S', 'D', 'S']
    b = ['S', 'D', 'D', 'D', 'S', 'D', 'S', 'D', 'D', 'S']
    assert he.cohen_kappa(a, b) == pytest.approx(sk.cohen_kappa_score(a, b))
    for table in ([8, 2, 1, 5], [3, 17, 15, 5], [0, 10, 10, 0], [4, 4, 4, 4]):
        ref = scipy_stats.fisher_exact([[table[0], table[1]], [table[2], table[3]]]).pvalue
        assert he.fisher_exact(*table) == pytest.approx(ref, rel=1e-6)
    lo, hi = he.wilson(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)


def test_human_eval_sheets_and_score_end_to_end(tmp_path):
    import csv
    import gzip
    import types
    he = _human_eval_module()
    rng = np.random.default_rng(0)
    root = tmp_path / 'artifacts'
    # H1 source: calibration groups (G=8) with partitioner labels on correct rollouts
    cal = root / 'calibration' / '0.5B'
    cal.mkdir(parents=True)
    with open(cal / 'calibration_groups.jsonl', 'w') as f:
        for g in range(20):
            ok = [1, 1, 1, 1, 0, 1, 0, 1]
            lab = [0, 0, 1, 1, -1, 2, -1, 0]
            f.write(json.dumps({'prompt_id': g, 'ground_truth': '1', 'problem': f'problem {g}',
                                'completions': [f'g{g} sol {i}' for i in range(8)], 'is_correct': ok,
                                'labels_chosen': lab}) + '\n')
    # H2 source: base eval + 3 vanilla seeds on 60 problems
    stem = 'math500_n32_p500'
    P, n = 60, 32
    ids = np.array([f'math500_{i}' for i in range(P)])
    base = (rng.random((P, n)) < 0.3).astype(np.int8)
    base[:, 0] = 1                                     # every problem solved by the base model
    bdir = root / 'baseline' / '0.5B' / 'eval'
    bdir.mkdir(parents=True)
    np.savez_compressed(bdir / f'{stem}.npz', correctness=base, prompt_ids=ids)
    with gzip.open(bdir / f'{stem}_completions.jsonl.gz', 'wt', encoding='utf-8') as f:
        for i in range(P):
            f.write(json.dumps({'prompt_id': ids[i], 'problem': f'p{i}', 'ground_truth': '1',
                                'correct': base[i].tolist(), 'completions': [f's{i}_{j}' for j in range(n)]}) + '\n')
    lost = set(range(0, 20))                           # vanilla never solves problems 0-19
    for seed in range(3):
        v = (rng.random((P, n)) < 0.4).astype(np.int8)
        v[:, 1] = 1
        v[sorted(lost)] = 0
        d = root / 'runs' / 'vanilla' / '0.5B' / f'seed{seed}' / 'eval'
        d.mkdir(parents=True)
        np.savez_compressed(d / f'{stem}.npz', correctness=v, prompt_ids=ids)
    he.cmd_sheets(types.SimpleNamespace(artifact_root=str(root), model_size='0.5B', out=None, n_pairs=20,
                                        n_exit=10, annotators='A,B', seed=0))
    key = json.loads((root / 'human_eval' / 'key.json').read_text())
    assert len(key['h1']) == 20 and sum(v['partitioner'] == 'same' for v in key['h1'].values()) == 10
    assert sum(v['group'] == 'exited' for v in key['h2'].values()) == 10
    assert all(int(v['prompt_id'].split('_')[1]) in lost for v in key['h2'].values() if v['group'] == 'exited')

    def fill(ann, flip):
        for fname, truth in (('h1_pairs.csv', lambda k: 'SAME' if key['h1'][k]['partitioner'] == 'same' else 'DIFFERENT'),
                             ('h2_solutions.csv', lambda k: 'FLAWED' if key['h2'][k]['group'] == 'exited' else 'VALID')):
            path = root / 'human_eval' / f'annotator_{ann}' / fname
            rows = list(csv.DictReader(open(path, newline='', encoding='utf-8-sig')))
            for i, r in enumerate(rows):
                r['label'] = truth(r['item_id'])
                if i < flip:
                    r['label'] = {'SAME': 'DIFFERENT', 'DIFFERENT': 'SAME', 'VALID': 'FLAWED', 'FLAWED': 'VALID'}[r['label']]
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    fill('A', 0)
    fill('B', 2)
    he.cmd_score(types.SimpleNamespace(artifact_root=str(root), sheets=None))
    res = json.loads((root / 'analysis' / 'human_eval.json').read_text())
    assert res['h1']['partitioner_accuracy'] == 1.0 and res['h1']['n_consensus'] == 18
    assert 0 < res['h1']['kappa'] < 1
    g = res['h2']['groups']
    assert g['exited']['rate'] == 0.0 and g['retained']['rate'] == 1.0 and res['h2']['fisher_p'] < 0.01


def test_generation_copy_matches_lora_and_leaves_model_untouched():
    torch = pytest.importorskip('torch')
    from rlvr.model_utils import generation_copy
    model = _tiny_model().eval()                      # init_lora_weights=False: adapters are non-zero
    ids = torch.tensor([[3, 9, 27, 4, 11, 60, 2]])
    with torch.no_grad():
        lora = model(input_ids=ids).logits
        with model.disable_adapter():
            base = model(input_ids=ids).logits
        with generation_copy(model) as gen:
            assert gen is not model and not hasattr(gen, 'peft_config')
            merged = gen(input_ids=ids).logits
        after = model(input_ids=ids).logits
    assert (lora - base).abs().max() > 1e-3            # the adapters really change the output
    assert torch.allclose(merged, lora, atol=1e-5)     # the copy carries them (fp32: exact up to float error)
    assert torch.equal(after, lora)                    # the training model is unchanged
    with generation_copy(model, enabled=False) as same:
        assert same is model
