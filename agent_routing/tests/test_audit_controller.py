"""Keep failed samples in evaluation and protect scientific dataset boundaries."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_verifiable import Advisors, Backend, CFG, row
from src.verifiable.data import identity, load_rows
from src.verifiable.experiment import policy_rollout, summary

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('audit_aime', SCRIPTS / 'runpod_aime_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_partial_and_empty_advice_are_observations_not_batch_failures():
    class PartialAdvisor(Advisors):
        def __init__(self, text):
            self.text = text

        def call(self, *args):
            return {'text': self.text, 'prompt_tokens': 1, 'completion_tokens': 2,
                    'truncated': bool(self.text), 'error': 'advisor_output_truncated' if self.text else 'advisor_empty_output'}

    for text in ('Partial derivation...', ''):
        backend = Backend()
        result = policy_rollout(row(), backend, PartialAdvisor(text), CFG, 42)
        assert result['correct'] and result['valid']
        assert result['costs'][1]['text'] == text
        assert result['costs'][1]['error']
        assert all(message['content'] for history in backend.inputs for message in history if message['role'] == 'tool')


def test_advisor_connection_failure_remains_fatal():
    class BrokenAdvisor:
        def call(self, *args):
            raise ConnectionError('offline')
    with pytest.raises(ConnectionError, match='offline'):
        policy_rollout(row(), Backend(), BrokenAdvisor(), CFG, 42)


@pytest.mark.parametrize('truncated,error', [(True, 'answer_truncated'), (False, 'answer_format')])
def test_invalid_committed_candidate_records_reason(truncated, error):
    class CommitBackend(Backend):
        def generate(self, *args, **kwargs):
            return {'text': 'COMMIT', 'truncated': False, 'prompt_tokens': 2, 'completion_tokens': 1}
    root = {'text': 'unfinished', 'valid': False, 'truncated': truncated}
    result = policy_rollout(row(), CommitBackend(), Advisors(), CFG, 42, root=root, history=[])
    assert not result['correct'] and not result['valid']
    assert result['error'] == error


def test_invalid_outputs_stay_in_summary_denominator():
    good = {'direct_correct': True, 'direct_valid': True, 'direct_truncated': False,
            'policy': {'correct': True, 'valid': True, 'calls': 0, 'error': None}, 'costs': []}
    bad = {'direct_correct': False, 'direct_valid': False, 'direct_truncated': True,
           'policy': {'correct': False, 'valid': False, 'calls': 1, 'error': 'answer_truncated'},
           'costs': [{'role': 'advisor', 'text': 'partial', 'truncated': True,
                      'prompt_tokens': 1, 'completion_tokens': 2, 'error': 'advisor_output_truncated'}]}
    metrics = summary([good, bad])
    assert metrics['n'] == 2
    assert metrics['independent_accuracy'] == metrics['policy_accuracy'] == .5
    assert metrics['direct_valid_rate'] == metrics['direct_truncated_rate'] == .5
    assert metrics['policy_error_counts'] == {'answer_truncated': 1}
    assert metrics['generation_diagnostics']['advisor']['truncated_n'] == 1


def test_context_limit_is_counted_invalid_without_retry_or_prompt_truncation():
    from src.verifiable.backend import ContextBudgetExceeded
    class TooLong:
        calls = 0
        def generate(self, *args, **kwargs):
            self.calls += 1
            raise ContextBudgetExceeded(9500, 1000, 10000)
    backend = TooLong()
    result = policy_rollout(row(), backend, Advisors(), CFG, 42)
    assert not result['correct'] and not result['valid']
    assert result['error'] == 'context_budget_exceeded'
    assert backend.calls == 1 and result['calls'] == 0


def test_complete_check_rejects_stale_aggregate_or_duplicate_expectations(tmp_path):
    record = {'question_hash': 'q1', 'direct_correct': False}
    (tmp_path / 'records.jsonl').write_text(json.dumps(record) + '\n')
    (tmp_path / 'summary.json').write_text('{"n": 1}')
    (tmp_path / 'questions').mkdir()
    (tmp_path / 'questions/q1.json').write_text(json.dumps({**record, 'direct_correct': True}))
    with pytest.raises(ValueError, match='checkpoints disagree'):
        baseline.verify_complete(tmp_path, ['q1'])
    with pytest.raises(ValueError, match='nonempty and unique'):
        baseline.verify_complete(tmp_path, ['q1', 'q1'])


def test_preflight_quality_is_warning_unless_explicitly_strict():
    record = {'direct_valid': False, 'policy': {'valid': False},
              'costs': [{'role': 'manager', 'truncated': True}]}
    observed = baseline.preflight_stats([record] * 3)
    assert observed['proceed'] and len(observed['warnings']) == 2
    assert not baseline.preflight_stats([record] * 3, strict=True)['proceed']


def test_math_reader_rejects_stale_content_identity(tmp_path):
    record = row().to_dict()
    record['metadata']['content_hash'] = identity('different question')
    file = tmp_path / 'data.jsonl'
    file.write_text(json.dumps(record) + '\n')
    with pytest.raises(ValueError, match='content hash'):
        load_rows(file)


def test_explicit_test_only_pool_never_becomes_train():
    from src.pipeline.stages import _split_rows
    rows = [row(i, split='test') for i in range(12)]
    train, dev, test = _split_rows(rows, 6, 2, 12, 42)
    assert train == dev == []
    assert len(test) == 12 and all(r.split == 'test' for r in test)
    rows = [row(i, split='') for i in range(12)]
    train, dev, test = _split_rows(rows, 6, 2, 3, 42)
    assert all(r.split == 'train' for r in train)
    assert all(r.split == 'dev' for r in dev)
    assert all(r.split == 'test' for r in test)


def test_gpqa_missing_exclusion_fails_closed(monkeypatch):
    from src.benchmarks.gpqa import _collect_subset_questions
    def unavailable(*args, **kwargs):
        raise OSError('gated or offline')
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=unavailable))
    with pytest.raises(RuntimeError, match='exclusion subset'):
        _collect_subset_questions('fake', ['gpqa_diamond'], None)


def test_gpqa_nested_configs_deduplicate(monkeypatch):
    from src.benchmarks.gpqa import load_gpqa
    def example(question):
        return {'Question': question, 'Correct Answer': 'x', 'Incorrect Answer 1': 'a',
                'Incorrect Answer 2': 'b', 'Incorrect Answer 3': 'c'}
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=lambda *a, **k: {'train': [example('shared')]}))
    records = load_gpqa(subsets='gpqa_diamond,gpqa_main')
    assert len(records) == 1


def test_json_teacher_parser_accepts_nested_objects_and_trailing_prose():
    from src.subagents.synthesize import _extract_first_json
    assert _extract_first_json('Here: {"a": {"b": "}"}}\nnext {"c": 3}') == {'a': {'b': '}'}}


def test_subagent_training_synthesis_rejects_test_rows_before_teacher_call(tmp_path):
    from src.subagents.synthesize import synthesize_subagent_data
    from src.subagents.schemas import AgentKind
    with pytest.raises(ValueError, match='development/test'):
        synthesize_subagent_data([row(split='test')], AgentKind.EXTRACTOR, None,
                                str(tmp_path / 'sft.jsonl'))
    assert not (tmp_path / 'sft.jsonl').exists()


def test_paired_report_rejects_changed_gold_for_same_question():
    from src.verifiable.reporting import aligned, paired_stats
    with pytest.raises(ValueError, match='metadata differs'):
        aligned([{'question_hash': 'x', 'ground_truth': '1'}],
                [{'question_hash': 'x', 'ground_truth': '2'}])
    with pytest.raises(ValueError, match='binary outcomes'):
        paired_stats([True], [float('nan')])


def test_recovered_torn_usage_never_becomes_complete_cost_accounting(tmp_path):
    from src.verifiable.reporting import stage_cost
    events = [{'attempt': 'one', 'event': 'usage_tail_recovered', 'discarded_bytes': 10},
              {'attempt': 'one', 'event': 'started'},
              {'attempt': 'one', 'event': 'completed', 'wall_seconds': 2},
              {'attempt': 'two', 'event': 'started'},
              {'attempt': 'two', 'event': 'completed', 'wall_seconds': 3}]
    (tmp_path / 'events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in events))
    (tmp_path / 'usage.jsonl').write_text(json.dumps({'role': 'manager', 'completion_tokens': 12}) + '\n')
    observed = stage_cost(tmp_path)
    assert observed['usage_incomplete'] and not observed['accounting_complete']
    assert observed['actual_generation_tokens'] == 12
    assert observed['observed_wall_seconds'] == 5


def test_paper_hashes_and_costs_ignore_tracking_backups(tmp_path, monkeypatch):
    from test_math_reporting import fixture
    from src.verifiable import reporting
    monkeypatch.setattr(reporting, 'plot', lambda *args: [])
    root = fixture(tmp_path)
    before, after = tmp_path / 'before', tmp_path / 'after'
    reporting.generate_report([root], before, demo=True)
    for directory in (root / 'tracking_exports/attempt',
                      root / 'round_1/sft/tracking_exports/attempt',
                      root / 'round_1/rl/wandb/run/files'):
        directory.mkdir(parents=True)
        (directory / 'usage.jsonl').write_text('{"role":"manager","completion_tokens":999999}\n')
        (directory / 'loop.json').write_text('{"snapshot":"not a new experiment"}')
        (directory / 'events.jsonl').write_text('{"attempt":"backup","event":"failed"}\n')
    reporting.generate_report([root], after, demo=True)
    old_hashes = json.loads((before / 'report_manifest.json').read_text())['input_sha256']
    new_hashes = json.loads((after / 'report_manifest.json').read_text())['input_sha256']
    assert old_hashes == new_hashes
    assert (before / 'costs.csv').read_bytes() == (after / 'costs.csv').read_bytes()
    assert (before / 'main_results.csv').read_bytes() == (after / 'main_results.csv').read_bytes()


def test_rsi_costs_include_failed_attempts_and_unfinished_stages(tmp_path):
    from src.verifiable.rsi import report
    stages = [tmp_path / 'dynamic/round_1/sft', tmp_path / 'static/round_1/sft',
              tmp_path / 'success/round_1/sft']
    (tmp_path / 'rsi_run.json').write_text(json.dumps({'rounds': 1,
        'plan': [{'stage': 'sft', 'output': str(path)} for path in stages]}))
    resumed, running, absent = stages
    resumed.mkdir(parents=True)
    running.mkdir(parents=True)
    events = [{'attempt': 'failed', 'event': 'started'},
              {'attempt': 'failed', 'event': 'failed', 'wall_seconds': 7},
              {'attempt': 'resumed', 'event': 'started'},
              {'attempt': 'resumed', 'event': 'completed', 'wall_seconds': 5}]
    (resumed / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    (resumed / 'usage.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in [
        {'role': 'manager', 'attempt': 'failed', 'actual_completion_tokens': 40},
        {'role': 'manager', 'attempt': 'resumed', 'actual_completion_tokens': 20}]))
    (resumed / '.rsi_complete.json').write_text('{"wall_seconds": 6}')
    (running / 'events.jsonl').write_text('{"attempt":"active","event":"started"}\n')
    (running / 'usage.jsonl').write_text('{"role":"manager","completion_tokens":11}\n')
    result = report(tmp_path)
    costs = {entry['stage']: entry for entry in result['stage_costs']}
    assert len(costs) == 3
    done = costs['dynamic/round_1/sft']
    assert done['observed_wall_seconds'] == 12 and done['attempts'] == 2
    assert done['actual_generation_tokens'] == 60
    assert done['last_successful_attempt_wall_seconds'] == done['wall_seconds'] == 6
    assert done['stage_complete'] and not done['accounting_complete']
    active = costs['static/round_1/sft']
    assert active['actual_generation_tokens'] == 11
    assert active['observed_wall_seconds'] is None and active['wall_seconds'] is None
    assert active['observed_activity'] and not active['stage_complete']
    missing = costs['success/round_1/sft']
    assert missing['actual_generation_tokens'] is None and missing['observed_wall_seconds'] is None
    assert not missing['observed_activity'] and not missing['accounting_complete']
    assert not result['complete'] and result['completed_stages'] == 1
    # A read-only report must also survive a writer killed mid-JSON record.
    broken = (running / 'usage.jsonl').read_text() + '{"role":'
    (running / 'usage.jsonl').write_text(broken)
    costs = {entry['stage']: entry for entry in report(tmp_path)['stage_costs']}
    assert costs['static/round_1/sft']['accounting_error'] == 'JSONDecodeError'
    assert costs['static/round_1/sft']['actual_generation_tokens'] is None
    assert costs['static/round_1/sft']['observed_activity']
    assert (running / 'usage.jsonl').read_text() == broken
