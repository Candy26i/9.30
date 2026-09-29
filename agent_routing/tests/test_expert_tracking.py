import json
from pathlib import Path

from test_math_wandb import FakeWandb
from src.verifiable.wandb_tracking import WandbTracker


def test_three_experts_share_parent_group_and_keep_role_and_evidence(tmp_path, monkeypatch):
    sdk = FakeWandb()
    monkeypatch.setenv('MARGENT_WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_ENTITY', 'test')
    monkeypatch.setenv('WANDB_PROJECT', 'math')
    monkeypatch.setenv('MARGENT_WANDB_TEXT', '0')
    monkeypatch.setattr('src.verifiable.wandb_tracking.import_module', lambda _: sdk)
    (tmp_path / 'expert_run.json').write_text(json.dumps({'config': {'seed': 42}, 'purpose': 'expert_sft_before_manager'}))
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'manifest.json').write_text('{"question_counts":{"train":128,"dev":32}}')
    (data / 'references.jsonl').write_text('{"solution":"PRIVATE_REFERENCE"}\n')
    parent = WandbTracker(tmp_path, 'expert_sft_controller', 'parent')
    parent.start()
    parent.snapshot_artifacts()
    files = sdk.runs[0].artifacts[-1].files
    assert 'data/manifest.json' in files and 'data/references.jsonl' not in files
    assert 'PRIVATE_REFERENCE' not in str(files)
    for role in ('extractor', 'reasoner', 'verifier'):
        root = tmp_path / 'training' / role
        root.mkdir(parents=True)
        (root / 'training_run.json').write_text(json.dumps({'role': role, 'config': {'max_steps': 16}}))
        (root / 'dev_metrics.json').write_text('{"eval_loss":1.25,"n":32}')
        tracker = WandbTracker(root, 'expert_sft', role)
        tracker.start()
        tracker.snapshot_artifacts()
        assert 'dev_metrics.json' in sdk.runs[-1].artifacts[-1].files
    assert len({run['group'] for run in sdk.calls}) == 1
    assert len({run['id'] for run in sdk.calls}) == 4
    assert [run['config']['expert_role'] for run in sdk.calls[1:]] == ['extractor', 'reasoner', 'verifier']
    monkeypatch.setenv('MARGENT_WANDB_TEXT', '1')
    opted = WandbTracker(tmp_path, 'expert_sft_controller', 'text')
    opted.start()
    opted.snapshot_artifacts()
    assert 'data/references.jsonl' in sdk.runs[-1].artifacts[-1].files


def test_teacher_synthesis_preserves_metadata_and_text_opt_in(tmp_path, monkeypatch):
    sdk = FakeWandb()
    monkeypatch.setenv('MARGENT_WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_ENTITY', 'test')
    monkeypatch.setenv('WANDB_PROJECT', 'math')
    monkeypatch.setenv('MARGENT_WANDB_TEXT', '0')
    monkeypatch.setattr('src.verifiable.wandb_tracking.import_module', lambda _: sdk)
    (tmp_path / 'synthesis_run.json').write_text(json.dumps({'config': {'provider': 'openai', 'model': 'gpt-4o'}}))
    (tmp_path / 'synthesis_status.json').write_text('{"complete":false,"accepted_tasks":1}')
    (tmp_path / 'teacher_responses.jsonl').write_text('{"text":"TEACHER_PRIVATE_TEXT"}\n')
    shard = tmp_path / 'calls/request/0000.json'
    shard.parent.mkdir(parents=True)
    shard.write_text('{"text":"TEACHER_PRIVATE_TEXT"}')
    tracker = WandbTracker(tmp_path, 'expert_teacher_synthesis', 'metadata')
    tracker.start()
    tracker.snapshot_artifacts()
    files = sdk.runs[-1].artifacts[-1].files
    assert sdk.calls[-1]['config']['model'] == 'gpt-4o'
    assert 'synthesis_run.json' in files and 'synthesis_status.json' in files
    assert 'teacher_responses.jsonl' not in files and 'calls/request/0000.json' not in files
    monkeypatch.setenv('MARGENT_WANDB_TEXT', '1')
    tracker = WandbTracker(tmp_path, 'expert_teacher_synthesis', 'text')
    tracker.start()
    tracker.snapshot_artifacts()
    files = sdk.runs[-1].artifacts[-1].files
    assert 'teacher_responses.jsonl' in files and 'calls/request/0000.json' in files
