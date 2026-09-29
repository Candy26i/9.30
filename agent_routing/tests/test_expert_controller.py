"""Exercise the expert-first lifecycle with local data and real tiny adapters."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from src.verifiable import expert_data, expert_train, experts
from src.verifiable.backend import HFBackend
from src.verifiable.protocol import KINDS
from src.verifiable.serve import load_expert_bundle
from test_expert_data import manager_data, record, write_jsonl
from test_expert_train import tiny_base, tiny_cpu, write_json


def setup_run(tmp_path):
    base = tiny_base(tmp_path / 'base')
    manager = manager_data(tmp_path)
    raw = tmp_path / 'raw.jsonl'
    write_jsonl(raw, [record(i) for i in range(120)])
    cfg = dict(base_model=str(base), base_model_revision=None, seed=42, train_size=3,
               dev_size=2, scan_limit=120, max_steps=1, save_steps=1, max_seq_len=1024,
               gradient_accumulation_steps=1, lora_rank=2, lora_alpha=4,
               lora_dropout=.05, bf16=False)
    path = tmp_path / 'expert_config.json'
    write_json(path, cfg)
    manager_cfg = json.loads((Path(__file__).parents[1] / 'configs/math_rsi_actions.json').read_text())
    manager_cfg.update(base_model=str(base), base_model_revision=None)
    manager_path = tmp_path / 'source_manager_config.json'
    write_json(manager_path, manager_cfg)
    return SimpleNamespace(config=str(path), manager_config=str(manager_path),
        manager_data_dir=str(manager), out=str(tmp_path / 'run'), raw_jsonl=str(raw), minutes=2, gpu='0')


def install_local_stages(monkeypatch, calls):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')  # Restore controller mutation after each test.
    monkeypatch.setattr(experts, 'gpu_preflight', lambda gpu: {'fixture': 'CPU', 'physical_gpu': gpu})
    def generate(self, messages, **kwargs):
        assert self.model.active_adapters and not self.model.training
        assert not any(parameter.requires_grad for parameter in self.model.parameters())
        return {'text': 'five', 'prompt_tokens': 5, 'completion_tokens': 1,
                'seconds': 0., 'truncated': False}
    monkeypatch.setattr(HFBackend, 'generate', generate)
    def execute(step, root, deadline, monitor):
        assert time.time() < deadline
        calls.append(step['name'])
        (root / 'logs' / (step['name'] + '.log')).write_text('local CPU integration stage\n')
        command = step['command']
        def arg(name):
            return command[command.index(name) + 1]
        if step['name'] == 'data':
            expert_data.build_expert_data(arg('--out'), arg('--manager-data-dir'),
                raw_jsonl=arg('--raw-jsonl'), train_size=int(arg('--train-size')),
                dev_size=int(arg('--dev-size')), scan_limit=int(arg('--scan-limit')),
                seed=int(arg('--seed')), resume=(root / 'data/manifest.json').exists())
        elif step['name'].startswith('sft_'):
            expert_train.train_expert(arg('--config'), arg('--data-dir'), arg('--role'),
                                      arg('--output'), resume='--resume' in command)
        elif step['name'] == 'reload_smoke':
            experts.smoke(arg('--bundle'), arg('--out'))
        else:
            pytest.fail('Expert controller attempted an unauthorized stage: ' + step['name'])
    monkeypatch.setattr(experts, 'run_child', execute)


@pytest.fixture
def completed_run(tmp_path, monkeypatch, tiny_cpu):
    args = setup_run(tmp_path)
    calls = []
    install_local_stages(monkeypatch, calls)
    report = experts.run(args)
    return args, Path(args.out), report, calls


def test_plan_trains_all_experts_before_reload_without_starting_manager_or_test(tmp_path):
    config = Path(__file__).parents[1] / 'configs/math_expert_sft_pilot.json'
    steps = experts.build_plan(config, tmp_path / 'manager-data', tmp_path / 'out', tmp_path / 'raw.jsonl')
    assert [step['name'] for step in steps] == ['data', 'sft_extractor', 'sft_reasoner', 'sft_verifier', 'reload_smoke']
    assert '--resume' not in steps[0]['command']
    for role, step in zip(KINDS, steps[1:4]):
        command = step['command']
        assert command[command.index('-m') + 1] == 'src.verifiable.expert_train'
        assert command[command.index('--role') + 1] == role
        assert Path(command[command.index('--output') + 1]).name == role
    modules = {s['command'][s['command'].index('-m') + 1] for s in steps}
    assert modules == {'src.verifiable.expert_data', 'src.verifiable.expert_train', 'src.verifiable.experts'}
    assert not any(token in {'evaluate', 'aime2026', '--checkpoint'} for s in steps for token in s['command'])


def test_data_child_gets_resume_only_after_published_manifest(tmp_path):
    (tmp_path / 'logs').mkdir()
    monitor = SimpleNamespace(update=lambda **kwargs: None)
    step = {'name': 'data', 'command': [sys.executable, '-c',
        "import sys; print('resume=' + str('--resume' in sys.argv))"]}
    experts.run_child(step, tmp_path, time.time() + 20, monitor)
    (tmp_path / 'data').mkdir()
    write_json(tmp_path / 'data/manifest.json', {'published': True})
    experts.run_child(step, tmp_path, time.time() + 20, monitor)
    assert (tmp_path / 'logs/data.log').read_text().splitlines() == ['resume=False', 'resume=True']
    assert '--resume' not in step['command'], 'Resume decoration must not mutate the persistent plan'


def test_failure_records_stage_and_persistent_deadline_cannot_restart_gpu(tmp_path, monkeypatch, tiny_cpu):
    args = setup_run(tmp_path)
    calls = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(experts, 'gpu_preflight', lambda gpu: {'fixture': 'CPU'})
    def fail(step, root, deadline, monitor):
        calls.append(step['name'])
        (root / 'logs' / 'data.log').write_text('specific dataset construction error\n')
        raise RuntimeError('data exited 9; see logs/data.log')
    monkeypatch.setattr(experts, 'run_child', fail)
    with pytest.raises(RuntimeError, match='data exited 9'):
        experts.run(args)
    root = Path(args.out)
    state = experts.read(root / 'expert_status.json')
    assert state['controller_status'] == 'failed' and state['failed_stage'] == 'data'
    assert not state['experts_complete']
    assert 'data exited 9' in (root / 'controller_traceback.txt').read_text()
    assert 'data exited 9' in (root / 'errors.log').read_text()
    budget_before = (root / 'budget.json').read_bytes()
    deadline = experts.read(root / 'budget.json')['deadline_unix']
    monkeypatch.setattr(experts, 'time', SimpleNamespace(time=lambda: deadline + 1))
    monkeypatch.setattr(experts, 'gpu_preflight', lambda gpu: pytest.fail('Expired run must not restart the GPU'))
    with pytest.raises(TimeoutError, match='Original expert budget'):
        experts.run(args)
    assert (root / 'budget.json').read_bytes() == budget_before and calls == ['data']
    state = experts.read(root / 'expert_status.json')
    assert state['controller_status'] == 'budget_exhausted' and not state['experts_complete']


def test_real_data_three_adapters_reload_export_and_idempotent_expired_resume(completed_run, monkeypatch):
    args, root, report, calls = completed_run
    assert calls == ['data', 'sft_extractor', 'sft_reasoner', 'sft_verifier', 'reload_smoke']
    assert report['experts_complete'] and not report['manager_started']
    assert report['controller_status'] == 'completed' and not report['pod_billing_stopped']
    bundle = load_expert_bundle(root / 'experts.json')
    assert set(bundle['roles']) == set(KINDS)
    for role in KINDS:
        summary = experts.read(root / 'training' / role / 'summary.json')
        assert summary['training_complete'] and summary['optimizer_steps'] == 1
        assert summary['base_model'] == bundle['base_model'] and summary['role'] == role
    smoke = experts.read(root / 'reload_smoke/summary.json')
    assert smoke['reload_complete'] and [x['role'] for x in smoke['requests']] == [*KINDS, 'extractor']
    generated_config = experts.read(root / 'manager_config.json')
    assert generated_config['advisor_expert_bundle'] == str(root / 'experts.json')
    assert generated_config == experts.manager_config(args.manager_config, root / 'experts.json')
    budget_before = (root / 'budget.json').read_bytes()
    deadline = experts.read(root / 'budget.json')['deadline_unix']
    monkeypatch.setattr(experts, 'time', SimpleNamespace(time=lambda: deadline + 1))
    write_json(root / 'expert_status.json', {'controller_status': 'failed', 'experts_complete': False})
    monkeypatch.setattr(experts, 'run_child', lambda *a, **k: pytest.fail('Completed run launched a child'))
    monkeypatch.setattr(experts, 'gpu_preflight', lambda *a: pytest.fail('Completed run touched GPU'))
    assert experts.run(args) == report
    assert experts.read(root / 'expert_status.json') == report
    assert (root / 'budget.json').read_bytes() == budget_before


def test_bundle_export_requires_completed_role_budget_and_unchanged_weights(completed_run):
    args, root, _, _ = completed_run
    cfg = experts.read(args.config)
    summary_path = root / 'training/reasoner/summary.json'
    original = summary_path.read_bytes()
    for changed in ({'training_complete': False}, {'role': 'extractor'}, {'optimizer_steps': 0}):
        write_json(summary_path, {**json.loads(original), **changed})
        with pytest.raises(ValueError):
            experts.export_bundle(root, cfg)
        summary_path.write_bytes(original)
    weights = root / 'training/reasoner/adapter_model.safetensors'
    weights.write_bytes(weights.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='adapter|artifact|export'):
        experts.export_bundle(root, cfg)


def test_first_bundle_export_rejects_adapter_configuration_changed_after_training(completed_run):
    args, root, _, _ = completed_run
    # Reproduce a change BEFORE first bundle creation, when no old bundle hash
    # can catch it: LoRA alpha changes the deployed function with identical weights.
    (root / 'experts.json').unlink()
    path = root / 'training/reasoner/adapter_config.json'
    config = experts.read(path)
    config['lora_alpha'] *= 2
    write_json(path, config)
    with pytest.raises(ValueError, match='artifact|export|fingerprint|training'):
        experts.export_bundle(root, experts.read(args.config))


def test_completed_resume_rejects_unbound_manager_config(completed_run):
    args, root, _, _ = completed_run
    config = experts.read(root / 'manager_config.json')
    config.pop('advisor_expert_bundle')
    write_json(root / 'manager_config.json', config)
    with pytest.raises(ValueError, match='Manager|manager|binding|config'):
        experts.run(args)
