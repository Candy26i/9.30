"""Regression evidence for training correctness and committed accounting."""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.manager.routing_anchor import tokenize_anchor_row
from src.verifiable.rsi_grpo import committed_step_directories, token_objective, validate_rl_config
from test_routing_anchor import _CharTokenizer, _base_prompt


class BoundaryTokenizer(_CharTokenizer):
    def __call__(self, text, add_special_tokens=False, **kwargs):
        # A real BPE tokenizer may merge the generation header's final byte
        # with the completion's first byte when the full render is encoded.
        text = text.replace('>A', 'Ω')
        return super().__call__(text, add_special_tokens)


def test_sft_boundary_preserves_exact_inference_prompt():
    tok = BoundaryTokenizer()
    row = dict(prompt=_base_prompt(), response='ANSWER_B')
    feature, stats = tokenize_anchor_row(row, tok, 4096, 'full')
    prompt = tok(tok.apply_chat_template(row['prompt'], add_generation_prompt=True))['input_ids']
    assert feature['input_ids'][:len(prompt)] == prompt
    assert feature['labels'][:len(prompt)] == [-100] * len(prompt)
    assert feature['labels'][len(prompt)] == ord('A')
    assert stats['supervised_tokens'] > 0


def test_subagent_rejects_partial_or_empty_training_targets():
    from src.subagents.train import _tokenize_subagent_sft
    rows = [dict(prompt=_base_prompt(), response='ANSWER_A')]
    with pytest.raises(ValueError, match='no silent truncation'):
        _tokenize_subagent_sft(rows, _CharTokenizer(), 20)
    with pytest.raises(ValueError, match='empty'):
        _tokenize_subagent_sft([dict(prompt=_base_prompt(), response='')], _CharTokenizer(), 4096)
    feature = _tokenize_subagent_sft(rows, BoundaryTokenizer(), 4096)[0]
    assert [x for x in feature['labels'] if x != -100][0] == ord('A')


def test_native_qwen_empty_thinking_prefix_is_retained_in_context():
    class NativeQwen(_CharTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
            result = super().apply_chat_template(messages, add_generation_prompt=add_generation_prompt, **kw)
            return result + '<think>\n\n</think>\n\n' if add_generation_prompt else result
    tok = NativeQwen()
    feature, stats = tokenize_anchor_row(dict(prompt=_base_prompt(), response='ANSWER_A'), tok, 4096, 'full')
    masked = ''.join(chr(x) for x, y in zip(feature['input_ids'], feature['labels']) if y == -100)
    supervised = ''.join(chr(x) for x in feature['labels'] if x != -100)
    assert masked.endswith('<think>\n\n</think>\n\n')
    assert supervised.startswith('ANSWER_A')


def test_truncated_anchor_is_excluded_with_explicit_count():
    from src.manager.routing_anchor import build_anchor_features
    features, stats = build_anchor_features([dict(prompt=_base_prompt(), response='ANSWER_A' * 100)], _CharTokenizer(), 90, 'full')
    assert not features
    assert stats['n_dropped_truncated'] == 1
    assert stats['n_dropped_no_target'] == 0
    assert stats['mean_supervised_tokens'] == 0


def test_zero_kl_coefficient_never_evaluates_overflowing_kl():
    import torch
    lp = torch.tensor([-1000.], requires_grad=True)
    loss = token_objective(lp, lp.detach(), torch.tensor([0.]), 1., beta=0.)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert lp.grad.item() == pytest.approx(-1.)


@pytest.mark.parametrize('field,value', [('rl_max_steps', 1.5), ('rl_temperature', float('nan')), ('rl_beta', float('nan')), ('rl_learning_rate', 0), ('rl_max_grad_norm', float('inf'))])
def test_invalid_rl_parameters_fail_before_model_load(field, value):
    cfg = dict(rl_max_steps=2, num_generations=4, rl_temperature=.8)
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_rl_config(cfg)


def _step(root, name, step):
    path = root / name
    path.mkdir()
    (path / 'step.json').write_text(json.dumps(dict(step=step)))
    return path


def test_orphaned_grpo_step_not_double_counted_on_resume(tmp_path):
    a = _step(tmp_path, 'step-00001-a', 1)
    _step(tmp_path, 'step-00002-orphan', 2)
    b = _step(tmp_path, 'step-00002-b', 2)
    _step(tmp_path, 'step-00003-orphan', 3)
    (tmp_path / 'resume.json').write_text(json.dumps(dict(directory=b.name, step=2,
        committed_directories=[a.name, b.name])))
    assert committed_step_directories(tmp_path) == [a, b]


def test_resume_pointer_step_disagreement_is_rejected(tmp_path):
    a = _step(tmp_path, 'step-00001-a', 1)
    (tmp_path / 'resume.json').write_text(json.dumps(dict(directory=a.name, step=2)))
    with pytest.raises(ValueError, match='disagree'):
        committed_step_directories(tmp_path)


def test_legacy_grpo_ignores_uncommitted_last_step(tmp_path):
    a = _step(tmp_path, 'step-00001-a', 1)
    _step(tmp_path, 'step-00002-orphan', 2)
    (tmp_path / 'resume.json').write_text(json.dumps(dict(directory=a.name, step=1)))
    assert committed_step_directories(tmp_path) == [a]


def test_old_manager_default_generation_batch_and_hub_id(monkeypatch):
    from src.manager.grpo_train import ManagerGRPOConfig, _resolve_checkpoint_source, _validate_generation_batch
    monkeypatch.setenv('WORLD_SIZE', '1')
    cfg = ManagerGRPOConfig(base_model='owner/base', rows=[], out_dir='out',
        extractor_adapter=None, reasoner_adapter=None, verifier_adapter=None)
    _validate_generation_batch(cfg)
    with pytest.raises(ValueError, match='divisible'):
        _validate_generation_batch(replace(cfg, gradient_accumulation_steps=2))
    assert _resolve_checkpoint_source('owner/adapter') == 'owner/adapter'
    assert _resolve_checkpoint_source('./local-adapter').endswith('/local-adapter')


def test_qwen35_multimodal_config_uses_text_causal_class():
    from src.subagents.train import load_text_causal_model
    with patch('transformers.AutoConfig.from_pretrained', return_value=SimpleNamespace(model_type='qwen3_5')) as config, \
         patch('transformers.Qwen3_5ForCausalLM.from_pretrained', return_value='text-model') as load:
        assert load_text_causal_model('model', revision='pinned', dtype='float32') == 'text-model'
        config.assert_called_once_with('model', revision='pinned')
        load.assert_called_once_with('model', revision='pinned', dtype='float32')


def test_real_subagent_lora_records_manifest_response_loss_and_weights(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file
    from test_rsi import tiny_checkpoint
    from src.subagents.train import train_subagent_sft, SFTConfig
    from src.utils.io import write_jsonl
    monkeypatch.setenv('MARGENT_WANDB_MODE', 'disabled')
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    base, _ = tiny_checkpoint(tmp_path)
    data = tmp_path / 'train.jsonl'
    write_jsonl(str(data), [dict(prompt=[dict(role='user', content='one')], response='two')])
    cfg = SFTConfig(base_model=str(base), train_jsonl=str(data), out_dir=str(tmp_path / 'out'),
        max_seq_len=4096, max_steps=1, gradient_accumulation_steps=1, lora_r=2, lora_alpha=4, bf16=False)
    train_subagent_sft(cfg)
    out = tmp_path / 'out'
    result = json.loads((out / 'training_metrics.json').read_text())
    assert result['optimizer_steps'] == 1
    assert result['train_loss'] > 0
    assert (out / 'training_run.json').is_file()
    assert (out / 'usage.jsonl').is_file()
    assert any(v.abs().sum() > 0 for k, v in load_file(str(out / 'adapter_model.safetensors')).items() if 'lora_B' in k)
    # Completed matching run is idempotent, not silently trained for a second epoch.
    with patch('src.subagents.train._train_subagent_sft', side_effect=AssertionError('should not retrain')):
        train_subagent_sft(cfg)


def test_replay_anchor_coefficient_does_not_scale_with_accumulation():
    import torch
    from src.manager.grpo_train import SFTAnchoredGRPOTrainer
    trainer = object.__new__(SFTAnchoredGRPOTrainer)
    trainer.sft_anchor_coef = .5
    trainer.args = SimpleNamespace(device='cpu', gradient_accumulation_steps=4)
    trainer.current_gradient_accumulation_steps = 4
    trainer.accelerator = SimpleNamespace(gather=lambda value: value)
    trainer._sft_anchor_loss_sum = 0.
    trainer._sft_anchor_loss_count = 0
    trainer._call_base_compute_loss = lambda *args, **kw: torch.tensor(1.)
    trainer._next_sft_anchor_batch = lambda: {'input_ids': torch.tensor([[1]]), 'labels': torch.tensor([[1]])}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(2.))
        def forward(self, **kwargs):
            return SimpleNamespace(loss=self.w)
    model = Model()
    loss = trainer.compute_loss(model, {})
    assert loss.item() == pytest.approx(1.25)
    loss.backward()
    assert model.w.grad.item() == pytest.approx(.125)


def test_context_rejected_grpo_group_has_no_fake_optimizer_update(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file
    from test_rsi import tiny_checkpoint
    from src.benchmarks.base import StandardRow
    from src.verifiable.rsi_grpo import train_grpo
    from src.utils.io import write_jsonl
    monkeypatch.setenv('MARGENT_WANDB_MODE', 'disabled')
    base, checkpoint = tiny_checkpoint(tmp_path)
    data = tmp_path / 'train.jsonl'
    write_jsonl(str(data), [StandardRow(0, 'tiny', 'math', 'one plus one', {}, '2', split='train', metadata={'answer_type': 'math'}).to_dict()])
    cfg = dict(base_model=str(base), seed=42, advisor_url='unused', advisor_max_tokens=8,
        max_context=4096, max_seq_len=4096, rl_temperature=.8, rl_max_steps=1, num_generations=2, lora_rank=2)
    empty = dict(correct=False, valid=False, calls=0, text='', error='context_budget_exceeded')
    with patch('src.verifiable.rsi_grpo.verify_advisor', return_value={}), \
         patch('src.verifiable.rsi_grpo.root_state', return_value=({'text': '', 'error': 'context_budget_exceeded'}, [])), \
         patch('src.verifiable.rsi_grpo.policy_rollout', return_value=empty):
        train_grpo(cfg, str(checkpoint), str(data), str(tmp_path / 'out'))
    summary = json.loads((tmp_path / 'out/training_metrics.json').read_text())
    assert summary['completed_groups'] == 1
    assert summary['optimizer_steps'] == 0
    assert summary['groups_without_sampled_tokens'] == 1
    step = json.loads((committed_step_directories(tmp_path / 'out')[0] / 'step.json').read_text())
    assert step['sample_count'] == 2
    assert step['reward_mean'] == 0
    assert step['manager_supervised_tokens'] == 0
    before, after = [load_file(str(p / 'adapter_model.safetensors')) for p in (checkpoint, tmp_path / 'out')]
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_real_legacy_manager_sft_logs_one_run_and_uses_response_only_mask(tmp_path, monkeypatch):
    from test_rsi import tiny_checkpoint
    from src.manager.evolve import ManagerSFTConfig, train_manager_sft
    from src.utils.io import write_jsonl
    monkeypatch.setenv('MARGENT_WANDB_MODE', 'disabled')
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    base, _ = tiny_checkpoint(tmp_path)
    data = tmp_path / 'train.jsonl'
    write_jsonl(str(data), [dict(prompt=[dict(role='user', content='one')], response='ANSWER_A')])
    cfg = ManagerSFTConfig(base_model=str(base), train_jsonl=str(data), out_dir=str(tmp_path / 'out'),
        max_seq_len=4096, max_steps=1, gradient_accumulation_steps=1, lora_r=2, lora_alpha=4, bf16=False)
    train_manager_sft(cfg)
    out = tmp_path / 'out'
    assert json.loads((out / 'training_metrics.json').read_text())['optimizer_steps'] == 1
    assert (out / 'training_run.json').is_file()
    assert (out / 'sft_data_report.json').is_file()
    with patch('src.manager.evolve._train_manager_sft', side_effect=AssertionError('should not retrain')):
        train_manager_sft(cfg)


def test_subagent_split_firewall_uses_question_identity_across_drafts():
    from src.subagents.train import validate_sft_splits
    train = dict(prompt=[dict(role='user', content='question draft A')], question_hash='same', split='train')
    dev = dict(prompt=[dict(role='user', content='question draft B')], question_hash='same', split='dev')
    with pytest.raises(ValueError, match='overlap'):
        validate_sft_splits([train], [dev])
    for split in ('test', 'dev', 'validation'):
        with pytest.raises(ValueError, match='train-only'):
            validate_sft_splits([{**train, 'split': split}])
    validate_sft_splits([train], [{**dev, 'question_hash': 'different'}])


def test_automatic_accumulation_preserves_legal_historical_batches(monkeypatch):
    from src.manager.grpo_train import ManagerGRPOConfig, _validate_generation_batch
    cfg = ManagerGRPOConfig(base_model='model', rows=[], out_dir='out', extractor_adapter=None,
                            reasoner_adapter=None, verifier_adapter=None)
    monkeypatch.setenv('WORLD_SIZE', '3')
    assert _validate_generation_batch(cfg) == 2  # old valid global batch 12
    assert _validate_generation_batch(replace(cfg, gradient_accumulation_steps=3)) == 3
    monkeypatch.setenv('WORLD_SIZE', '1')
    assert _validate_generation_batch(cfg) == 3  # old batch 4 cannot divide G=6
    with pytest.raises(ValueError, match='divisible'):
        _validate_generation_batch(replace(cfg, gradient_accumulation_steps=2))


def test_training_device_honors_local_rank_before_weight_loading(monkeypatch):
    from src.subagents.train import training_device
    monkeypatch.setenv('LOCAL_RANK', '2')
    with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.set_device') as select:
        assert training_device() == 'cuda'
        select.assert_called_once_with(2)
    with patch('torch.cuda.is_available', return_value=False), patch('torch.cuda.set_device') as select:
        assert training_device() == 'cpu'
        select.assert_not_called()
