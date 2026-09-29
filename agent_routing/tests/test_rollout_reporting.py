import importlib.util
import json
from pathlib import Path

from src.verifiable.rollout_reporting import reward_metrics, rollout_record


def test_reward_diagnostics_keep_failed_samples():
    samples = [dict(reward=0., outcome=dict(valid=False, calls=1)),
               dict(reward=1., outcome=dict(valid=True, calls=0))]
    stats = reward_metrics(samples, [-1., 1.])
    assert stats['reward_mean'] == .5
    assert stats['reward_std'] == .5
    assert stats['invalid_rate'] == .5
    assert stats['advantage_abs_mean'] == 1.
    assert stats['mixed_reward_group']
    assert [s['reward'] for s in samples] == [0., 1.]


def test_failure_report_distinguishes_action_from_answer():
    root = dict(text='root', valid=True, truncated=False)
    sample = dict(reward=0., turns=[dict(kind='decision'), dict(kind='revision')],
        outcome=dict(valid=False, text='unfinished', costs=[
            dict(role='manager', text='CALL', truncated=False),
            dict(role='advisor', text='hint'),
            dict(role='manager', text='unfinished', truncated=True)]))
    result = rollout_record(1, 0, root, sample, -1.)
    assert result['failure_type'] == 'revision_truncated'
    assert result['manager_outputs'][1]['text'] == 'unfinished'
    sample['outcome']['error'] = 'Unclosed tool call'
    assert rollout_record(1, 0, root, sample, -1.)['failure_type'] == 'decision_protocol'
    sample['outcome']['error'] = 'truncated_decision'
    assert rollout_record(1, 0, root, sample, -1.)['failure_type'] == 'decision_truncated'
    sample['outcome']['error'] = 'answer_truncated'
    assert rollout_record(1, 0, root, sample, -1.)['failure_type'] == 'revision_truncated'
    sample['outcome']['error'] = 'answer_format'
    assert rollout_record(1, 0, root, sample, -1.)['failure_type'] == 'revision_answer_format'
    sample['outcome']['error'] = 'context_budget_exceeded'
    assert rollout_record(1, 0, root, sample, -1.)['failure_type'] == 'context_budget_exceeded'


def test_review_uploads_real_offline_tables_without_changing_source(tmp_path, monkeypatch):
    import wandb
    spec = importlib.util.spec_from_file_location('review_smoke', Path(__file__).parents[1] / 'scripts/review_rsi_smoke.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv('MARGENT_WANDB_MODE', 'offline')
    monkeypatch.setenv('MARGENT_WANDB_TEXT', '1')
    monkeypatch.setenv('WANDB_ENTITY', 'offline-test')
    monkeypatch.setenv('WANDB_PROJECT', 'offline-test')
    source, target = tmp_path / 'source', tmp_path / 'review'
    def write(name, value):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write('config.json', {'seed': 42})
    write('smoke_status.json', {'status': 'failed', 'stage': 'validate'})
    for stage in ('collection', 'sft', 'grpo', 'after_grpo', 'next_sft'):
        write(stage + '/status.json', {'status': 'completed'})
        write(stage + '/training_metrics.json', {'optimizer_steps': 1})
    write('collection/sft.jsonl', {})
    write('after_grpo/summary.json', {'n': 1, 'policy_accuracy': 1.})
    write('grpo/step-00001/step.json', {'step': 1, 'advantages': [-1., 1.], 'loss': .01})
    write('grpo/step-00001/rollouts.json', {'root': {'text': 'root', 'valid': True}, 'trajectories': [
        {'reward': 0., 'outcome': {'valid': False, 'text': 'bad', 'error': 'Unclosed tool call'}},
        {'reward': 1., 'outcome': {'valid': True, 'text': 'good'}}]})
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    runs = []
    original_init = wandb.init
    def init(*a, **kw):
        run = original_init(*a, **kw)
        runs.append(run)
        return run
    monkeypatch.setattr(wandb, 'init', init)
    result = script.review(source, target)
    assert result['recorded_execution_completed']
    assert result['original_status']['status'] == 'failed'
    assert result['grpo_groups'][0]['reward_mean'] == .5
    assert result['failure_counts'] == {'decision_protocol': 1}
    assert result['reward_signal_present']
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert json.loads((target / 'status.json').read_text())['status'] == 'completed'
    tables = list(target.rglob('*.table.json'))
    rows = [json.loads(p.read_text()) for p in tables]
    rollout_tables = [t for t in rows if 'failure_type' in t['columns']]
    assert rollout_tables
    values = [row[t['columns'].index('reward')] for t in rollout_tables for row in t['data']]
    assert 0. in values and 1. in values
    assert (target / 'evidence/grpo/step-00001/rollouts.json').read_bytes() == before[Path('grpo/step-00001/rollouts.json')]
