import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('aime_baseline', SCRIPTS / 'runpod_aime_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_child_interruption_preserves_completed_question_and_restarts(tmp_path):
    questions = tmp_path / 'questions'
    questions.mkdir()
    pid = tmp_path / 'pid'
    code = """import json,os,time
from pathlib import Path
p=Path(%r)
(p/'pid').write_text(str(os.getpid()))
(p/'questions/q1.json').write_text(json.dumps({'question_hash':'q1'}))
time.sleep(60)
""" % str(tmp_path)
    assert baseline.run_child([sys.executable, '-c', code], tmp_path, tmp_path / 'log',
                              os.environ.copy(), time.time()+10, interrupt_after_one=True)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)
    saved = (questions / 'q1.json').read_bytes()
    code = """import json
from pathlib import Path
p=Path(%r)
if not (p/'questions/q1.json').exists(): raise RuntimeError('lost checkpoint')
(p/'questions/q2.json').write_text(json.dumps({'question_hash':'q2'}))
""" % str(tmp_path)
    baseline.run_child([sys.executable, '-c', code], tmp_path, tmp_path / 'log',
                       os.environ.copy(), time.time()+10)
    assert (questions / 'q1.json').read_bytes() == saved
    assert len(baseline.shards(tmp_path)) == 2


def test_child_deadline_kills_owned_process(tmp_path):
    with pytest.raises(TimeoutError):
        baseline.run_child([sys.executable, '-c', 'import time; time.sleep(60)'], tmp_path,
                           tmp_path / 'log', os.environ.copy(), time.time()+.1)


def test_partial_benchmark_never_looks_complete(tmp_path):
    (tmp_path / 'records.jsonl').write_text(json.dumps({'question_hash':'q1'})+'\n')
    (tmp_path / 'summary.json').write_text(json.dumps({'n':1}))
    with pytest.raises(ValueError, match='every expected question'):
        baseline.verify_complete(tmp_path, ['q1', 'q2'])
    assert len(baseline.verify_complete(tmp_path, ['q1'])) == 1


def test_preflight_does_not_gate_on_accuracy_or_one_bad_sample():
    valid = dict(direct_valid=True, direct_correct=False, policy={'valid':True,'correct':False}, costs=[])
    bad = dict(direct_valid=False, direct_correct=False, policy={'valid':False,'correct':False}, costs=[])
    assert baseline.preflight_stats([valid, valid, bad])['proceed']
    assert baseline.preflight_stats([valid, bad, bad])['proceed']
    assert not baseline.preflight_stats([valid, bad, bad], strict=True)['proceed']
    truncated = {**valid, 'costs':[dict(role='manager', truncated=True)]}
    assert baseline.preflight_stats([valid, truncated, truncated])['proceed']
    assert not baseline.preflight_stats([valid, truncated, truncated], strict=True)['proceed']


def test_benchmark_children_share_controller_wandb_group(tmp_path, monkeypatch):
    from test_math_wandb import FakeWandb
    from src.verifiable.wandb_tracking import WandbTracker
    fake = FakeWandb()
    monkeypatch.setenv('MARGENT_WANDB_MODE', 'online')
    monkeypatch.setenv('MARGENT_WANDB_TEXT', '0')
    monkeypatch.setenv('WANDB_ENTITY', 'test')
    monkeypatch.setattr('src.verifiable.wandb_tracking.import_module', lambda name: fake)
    (tmp_path / 'benchmark_run.json').write_text(json.dumps({'config':{'seed':42}}))
    child = tmp_path / 'aime2026'
    child.mkdir()
    parent = WandbTracker(tmp_path, 'aime_baseline', 'one')
    parent.start()
    stage = WandbTracker(child, 'evaluate', 'two')
    stage.start()
    assert json.loads((tmp_path/'wandb_run.json').read_text())['group'] == json.loads((child/'wandb_run.json').read_text())['group']
    parent.finish('completed')
    stage.finish('completed')


def test_resume_preserves_deadline_and_rejects_changed_inputs(tmp_path, monkeypatch):
    from src.benchmarks.base import StandardRow
    def rows(path, required_split):
        return [StandardRow(i, 'tiny', 'math', f'{required_split} question {i}', {}, '2', split=required_split)
                for i in range(3 if required_split == 'dev' else 30)]
    monkeypatch.setattr('src.verifiable.data.verify_manifest', lambda path: {'locked': True})
    monkeypatch.setattr('src.verifiable.data.load_rows', rows)
    output = tmp_path / 'run'
    output.mkdir()
    args = SimpleNamespace(out=str(output), data_dir=str(tmp_path),
        config=str(SCRIPTS.parent / 'configs/math_rsi_actions.json'), port=8003,
        manager_gpu='1', advisor_gpu='0', minutes=120)
    first = baseline.initialize(args)
    # Expired budgets are preserved too; restarting does not buy more runtime.
    (output / 'budget.json').write_text(json.dumps({'deadline_unix': 1}))
    assert baseline.initialize(args)[-1] == 1
    assert len(first[3]) == 3 and len(first[4]) == 30
    args.minutes = 60
    with pytest.raises(ValueError, match='settings changed'):
        baseline.initialize(args)
