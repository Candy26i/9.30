"""Offline teacher preparation through real tiny three-role expert training."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from src.teachers.base import TeacherClient, TeacherResponse
from src.verifiable import expert_synthesis, expert_train, experts
from src.verifiable.backend import HFBackend
from src.verifiable.protocol import DIRECT_SYSTEM, KINDS
from src.verifiable.serve import load_expert_bundle
from test_expert_data import manager_data, record, write_jsonl
from test_expert_train import make_data, tiny_base, tiny_cpu, write_json


class RecordedTeacher(TeacherClient):
    """Deterministic TeacherClient implementation with no network capability."""
    provider = 'openai'

    def __init__(self):
        super().__init__('gpt-4o')
        self.calls = []

    def chat(self, messages, temperature=.2, max_tokens=2048):
        self.calls.append(messages)
        system = messages[0]['content']
        assert 'OPAQUE_REFERENCE' not in json.dumps(messages)
        if system == DIRECT_SYSTEM:
            task = hashlib.sha256(messages[-1]['content'].encode()).hexdigest()[:8]
            text = f'Task {task}, sample {len(self.calls)}: add the stated x=2 and y=3 to obtain 5.\nFINAL_ANSWER: \\boxed{{5}}'
        elif 'Check whether the supplied derivation' in system:
            text = 'Verdict: correct\nEvidence: The given values add to five.\nCorrection: None needed'
        elif 'Extract givens' in system:
            text = 'The variables are x and y. The question gives x=2 and y=3 and asks for their sum.'
        else:
            text = 'Use the stated values and add the two quantities to find their sum.'
        return TeacherResponse(text, self.provider, self.model, raw={
            'model': 'gpt-4o-fixture-snapshot', 'id': f'fake-{len(self.calls)}', 'finish_reason': 'stop',
            'system_fingerprint': 'offline-fixture', 'usage': {'prompt_tokens': 24, 'completion_tokens': 16, 'total_tokens': 40},
            'provider_usage': {'total_tokens': 40}, 'latency_seconds': .01, 'request_attempts': 1})


@pytest.fixture
def teacher_source(tmp_path, monkeypatch):
    from src.teachers import base
    monkeypatch.setenv('MARGENT_WANDB_MODE', 'disabled')
    monkeypatch.setattr(base, 'build_teacher_client', lambda *a, **k: pytest.fail('Tests must never construct an API client'))
    manager = manager_data(tmp_path)
    raw = tmp_path / 'raw.jsonl'
    source_rows = [{**record(i), 'solution': f'OPAQUE_REFERENCE_{i}: this source solution is not teacher input.'}
                   for i in range(120)]
    write_jsonl(raw, source_rows)
    config = {'provider': 'openai', 'model': 'gpt-4o', 'train_size': 3, 'dev_size': 2,
              'scan_limit': 120, 'max_calls': 60, 'max_retries': 0}
    synthesis = tmp_path / 'synthesis'
    expert_synthesis.prepare(synthesis, manager, config, raw_jsonl=raw)
    teacher = RecordedTeacher()
    status = expert_synthesis.generate(synthesis, config, teacher=teacher)
    assert status['complete'] and len(teacher.calls) == 30
    manifest = expert_synthesis.finalize(synthesis)
    return SimpleNamespace(data=synthesis / 'data', manager=manager, synthesis=synthesis,
                           manifest=manifest, teacher=teacher, source_config=config)


def training_args(tmp_path, source, base):
    config = {'base_model': str(base), 'base_model_revision': None, 'seed': 42,
        'expected_teacher': {'provider': 'openai', 'model': 'gpt-4o'},
        'train_size': 3, 'dev_size': 2, 'max_steps': 1, 'save_steps': 1, 'max_seq_len': 1024,
        'gradient_accumulation_steps': 1, 'lora_rank': 2, 'lora_alpha': 4, 'bf16': False}
    # Deliberately omit expert_data_mode: the controller must default to teacher.
    path = tmp_path / 'training_config.json'
    write_json(path, config)
    manager_config = experts.read(Path(__file__).parents[1] / 'configs/math_rsi_actions.json')
    manager_config.update(base_model=str(base), base_model_revision=None)
    manager_path = tmp_path / 'manager_config_source.json'
    write_json(manager_path, manager_config)
    return SimpleNamespace(config=str(path), manager_config=str(manager_path), manager_data_dir=str(source.manager),
        expert_data_dir=str(source.data), out=str(tmp_path / 'training_run'), raw_jsonl=None, minutes=2, gpu='0')


def prevent_training(monkeypatch):
    monkeypatch.setattr(experts, 'gpu_preflight', lambda *a: pytest.fail('Invalid data must not touch GPU'))
    monkeypatch.setattr(experts, 'run_child', lambda *a: pytest.fail('Invalid data must not launch a child'))


def test_default_controller_requires_prepared_teacher_without_gpu(tmp_path, monkeypatch):
    manager = manager_data(tmp_path)
    missing = SimpleNamespace(data=tmp_path / 'not_prepared', manager=manager)
    args = training_args(tmp_path, missing, tmp_path)
    prevent_training(monkeypatch)
    with pytest.raises(FileNotFoundError):
        experts.run(args)
    assert not Path(args.out).exists()
    legacy = make_data(tmp_path / 'legacy')
    args.expert_data_dir = str(legacy)
    with pytest.raises(ValueError, match='requires completed teacher_synthetic'):
        experts.run(args)
    assert not Path(args.out).exists()


def test_teacher_configuration_and_manager_pool_are_checked_before_gpu(tmp_path, monkeypatch, teacher_source):
    args = training_args(tmp_path, teacher_source, tmp_path)
    prevent_training(monkeypatch)
    cfg = experts.read(args.config)
    write_json(Path(args.config), {**cfg, 'expected_teacher': {'provider': 'openai', 'model': 'wrong-teacher'}})
    with pytest.raises(ValueError, match='Teacher identity'):
        experts.run(args)
    write_json(Path(args.config), cfg)
    # A valid but changed Manager manifest must not reuse exclusions from the old pool.
    manager_data(tmp_path, questions=[f'A different locked pool sample {i}: unique policy version delta.' for i in range(4)])
    with pytest.raises(ValueError, match='different Manager/test pool'):
        experts.run(args)
    assert not Path(args.out).exists()


def test_prepare_teacher_copy_preserves_every_provenance_file_and_checks_reuse(tmp_path, teacher_source):
    args = training_args(tmp_path, teacher_source, tmp_path)
    destination = tmp_path / 'copied'
    manifest = experts.prepare_data(teacher_source.data, destination, teacher_source.manager, args.config)
    for name in (*manifest['sha256'], 'manifest.json'):
        assert (destination / name).read_bytes() == (teacher_source.data / name).read_bytes(), name
    assert experts.prepare_data(teacher_source.data, destination, teacher_source.manager, args.config) == manifest
    (destination / 'teacher_responses.jsonl').write_text('tampered evidence')
    with pytest.raises(ValueError, match='fingerprint'):
        experts.prepare_data(teacher_source.data, destination, teacher_source.manager, args.config)


def test_copy_race_cannot_publish_changed_evidence(tmp_path, monkeypatch, teacher_source):
    destination = tmp_path / 'copied'
    original = experts.shutil.copyfile
    def corrupt_during_copy(source, target):
        result = original(source, target)
        if Path(target).name == 'teacher_responses.jsonl':
            with Path(target).open('a') as stream:
                stream.write('changed in transit\n')
        return result
    monkeypatch.setattr(experts.shutil, 'copyfile', corrupt_during_copy)
    with pytest.raises(ValueError, match='fingerprint'):
        experts.prepare_data(teacher_source.data, destination, teacher_source.manager)
    assert not destination.exists()
    assert not list(tmp_path.glob('.copied.copy-*'))


def test_teacher_plan_has_no_generation_manager_or_benchmark_stage(tmp_path, teacher_source):
    args = training_args(tmp_path, teacher_source, tmp_path)
    plan = experts.build_plan(args.config, args.manager_data_dir, args.out, expert_data_dir=args.expert_data_dir)
    assert [step['name'] for step in plan] == ['data', 'sft_extractor', 'sft_reasoner', 'sft_verifier', 'reload_smoke']
    assert 'prepare-data' in plan[0]['command']
    assert not any('src.verifiable.expert_synthesis' in step['command'] for step in plan)
    with pytest.raises(ValueError, match='Prepare raw Numina'):
        experts.build_plan(args.config, args.manager_data_dir, args.out, raw_jsonl='raw.jsonl')


def test_real_teacher_three_role_sft_reload_and_completed_resume(tmp_path, monkeypatch, tiny_cpu, teacher_source):
    base = tiny_base(tmp_path / 'base')
    args = training_args(tmp_path, teacher_source, base)
    calls, gpu_checks = [], []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(experts, 'gpu_preflight', lambda gpu: gpu_checks.append(gpu) or {'fixture': 'CPU'})
    def generate(self, messages, **kwargs):
        assert self.model.active_adapters and not self.model.training
        assert not any(p.requires_grad for p in self.model.parameters())
        return {'text': 'The stated values can be added.', 'prompt_tokens': 8, 'completion_tokens': 6,
                'seconds': 0., 'truncated': False}
    monkeypatch.setattr(HFBackend, 'generate', generate)
    def execute(step, root, deadline, monitor):
        assert time.time() < deadline
        calls.append(step['name'])
        (root / 'logs' / (step['name'] + '.log')).write_text('local teacher CPU integration\n')
        command = step['command']
        def arg(name):
            return command[command.index(name) + 1]
        if step['name'] == 'data':
            assert gpu_checks == [], 'Teacher copying and validation must precede GPU allocation'
            experts.prepare_data(arg('--source'), arg('--out'), arg('--manager-data-dir'), arg('--config'))
        elif step['name'].startswith('sft_'):
            expert_train.train_expert(arg('--config'), arg('--data-dir'), arg('--role'), arg('--output'), resume=True)
        elif step['name'] == 'reload_smoke':
            experts.smoke(arg('--bundle'), arg('--out'))
        else:
            pytest.fail('Unexpected stage: ' + step['name'])
    monkeypatch.setattr(experts, 'run_child', execute)
    report = experts.run(args)
    root = Path(args.out)
    assert report['experts_complete'] and not report['manager_started']
    assert report['supervision'] == 'teacher_synthetic'
    assert report['teacher'] == {'provider': 'openai', 'model': 'gpt-4o', 'revision': None}
    assert calls == ['data', 'sft_extractor', 'sft_reasoner', 'sft_verifier', 'reload_smoke']
    assert gpu_checks == ['0'] and len(teacher_source.teacher.calls) == 30
    signature = experts.read(root / 'expert_run.json')['teacher_data_source']
    assert signature['manifest_sha256'] == experts.digest(root / 'data/manifest.json')
    assert signature['isolation']['checked']
    bundle = load_expert_bundle(root / 'experts.json')
    for role in KINDS:
        summary = report['roles'][role]
        assert summary['training_complete'] and summary['optimizer_steps'] == 1
        assert summary['supervision'] == 'teacher_synthetic' and summary['teacher_actual_models'] == ['gpt-4o-fixture-snapshot']
        assert summary['data_report']['train']['weak_supervision_rows'] == 0
        assert summary['data_report']['train']['teacher_generated_rows'] == (6 if role == 'verifier' else 3)
        assert Path(bundle['roles'][role]['checkpoint']).is_dir()
    for name in (*teacher_source.manifest['sha256'], 'manifest.json'):
        assert (root / 'data' / name).read_bytes() == (teacher_source.data / name).read_bytes()
    budget_before = (root / 'budget.json').read_bytes()
    deadline = experts.read(root / 'budget.json')['deadline_unix']
    monkeypatch.setattr(experts, 'time', SimpleNamespace(time=lambda: deadline + 1))
    prevent_training(monkeypatch)
    assert experts.run(args) == report
    assert (root / 'budget.json').read_bytes() == budget_before
    assert len(teacher_source.teacher.calls) == 30
