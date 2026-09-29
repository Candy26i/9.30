"""Offline teacher synthesis: no provider, model download, or credentials needed."""
import json
from types import SimpleNamespace

import pytest

from src.verifiable import expert_synthesis as synth
from src.verifiable.expert_train import read_dataset
from src.verifiable.protocol import KINDS
from test_expert_data import manager_data, record, write_jsonl, read_jsonl


class DummyMonitor:
    usages = []

    def __init__(self, *args, **kwargs):
        self.state = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, **kwargs):
        pass

    def summary(self, *args, **kwargs):
        pass

    def event(self, *args, **kwargs):
        pass

    def generation(self, *args, **kwargs):
        pass

    def usage(self, *args, **kwargs):
        self.usages.append((args, kwargs))


@pytest.fixture(autouse=True)
def offline_monitor(monkeypatch):
    DummyMonitor.usages = []
    monkeypatch.setattr(synth, 'Monitor', DummyMonitor)


class FakeTeacher:
    provider = 'openai'
    model = 'gpt-4o'

    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature, max_tokens):
        self.calls.append(dict(messages=messages, temperature=temperature, max_tokens=max_tokens))
        system = messages[0]['content']
        if 'independently' in system:
            text = f'Compute the sum 2+3=5. Independent draft {len(self.calls)}.\nFINAL_ANSWER: \\boxed{{5}}'
        elif 'Extract givens' in system:
            text = 'Givens: x=2 and y=3. Target: their sum. Distinguish stated quantities from deductions.'
        elif 'Develop a useful' in system:
            text = 'Substitute the stated values into the sum x+y, then simplify the resulting arithmetic expression.'
        else:
            text = 'Verdict: correct\nEvidence: The displayed addition is consistent.\nCorrection: None needed'
        return SimpleNamespace(text=text, provider=self.provider, model=self.model, raw={
            'id': f'fake-{len(self.calls)}', 'model': 'gpt-4o-fake-snapshot',
            'usage': {'prompt_tokens': 101, 'completion_tokens': 37, 'total_tokens': 138},
            'finish_reason': 'stop', 'latency_seconds': .1, 'request_attempts': 1})


def make_run(tmp_path, **options):
    manager = manager_data(tmp_path)
    raw = tmp_path / 'teacher_raw.jsonl'
    rows = [record(i) for i in range(80)]
    for row in rows:
        row.update(solution='SECRET_REFERENCE_NEVER_IN_TEACHER_INPUT', answer='99112233')
    write_jsonl(raw, rows)
    cfg = synth.load_config(dict(train_size=3, dev_size=2, scan_limit=80, **{'max_calls': 60, **options}))
    root = tmp_path / 'synthesis'
    synth.prepare(root, manager, cfg, raw_jsonl=raw)
    return root, manager, raw, cfg


def response_for(request, teacher=None):
    teacher = teacher or FakeTeacher()
    result = teacher.chat(request['messages'], request['temperature'], request['max_tokens'])
    return {'request_id': request['request_id'], 'response': {
        'text': result.text, 'provider': result.provider, 'model': result.model,
        'actual_model': result.raw['model'], 'finish_reason': result.raw['finish_reason'],
        'request_id': result.raw['id'], 'usage': result.raw['usage'],
        'latency_seconds': result.raw['latency_seconds'], 'request_attempts': 1}}


def test_prepare_freezes_gt_blind_pool_without_reference_arithmetic_gate(tmp_path, monkeypatch):
    from src.teachers import base
    monkeypatch.setattr(base, 'build_teacher_client', lambda *a, **k: pytest.fail('No client needed'))
    root, manager, raw, cfg = make_run(tmp_path)
    pool = read_jsonl(root / 'pool.jsonl')
    assert len(pool) == 5
    assert all('solution' not in p and 'ground_truth' not in p for p in pool)
    requests = read_jsonl(root / 'teacher_requests.jsonl')
    assert len(requests) == 20
    for request in requests:
        assert request['request_id'] == synth._hash({k: v for k, v in request.items() if k != 'request_id'})
        assert request['prompt_sha256'] == synth._hash(request['messages'])
        assert 'SECRET_REFERENCE' not in json.dumps(request['messages'])
        assert '99112233' not in json.dumps(request['messages'])
    run = synth.prepare(root, manager, cfg, raw_jsonl=raw, resume=True)
    assert run['question_counts'] == {'train': 3, 'dev': 2}
    assert 'stochastic' in run['sampling_reproducibility']
    assert 'teachers/openai_client.py' in run['code_sha256']
    with pytest.raises(ValueError, match='incomplete'):
        synth.finalize(root)


def test_complete_teacher_data_passes_real_training_loader_and_resumes(tmp_path):
    root, _, _, cfg = make_run(tmp_path)
    teacher = FakeTeacher()
    status = synth.generate(root, cfg, teacher=teacher)
    assert status['complete'] and status['accepted'] == 30 and len(teacher.calls) == 30
    assert status['budget']['observed_provider_request_attempts'] == 30
    assert all('SECRET_REFERENCE' not in json.dumps(c['messages']) for c in teacher.calls)
    manifest = synth.finalize(root)
    assert manifest['teacher'] == {'provider': 'openai', 'model': 'gpt-4o', 'revision': None}
    assert manifest['supervision'] == 'teacher_synthetic'
    assert len(manifest['sha256']) == 10
    for role in KINDS:
        dataset = read_dataset(root / 'data', role)
        assert dataset
        for split, n in [('train', 3), ('dev', 2)]:
            rows = read_jsonl(root / 'data' / role / f'{split}.jsonl')
            assert len(rows) == n * (2 if role == 'verifier' else 1)
            assert all(r['teacher_actual_model'] == 'gpt-4o-fake-snapshot' and not r['reviewed'] for r in rows)
            if role == 'verifier':
                assert all(r['verdict'] == 'correct' and not r['candidate_terminal_diagnostic']['terminal_correct'] for r in rows)
    assert synth.finalize(root) == manifest
    assert synth.generate(root, cfg, teacher=teacher, resume=True)['complete']
    assert len(teacher.calls) == 30


def test_import_exports_followup_verifier_then_publishes_no_api(tmp_path):
    root, _, _, _ = make_run(tmp_path)
    first = tmp_path / 'first.jsonl'
    teacher = FakeTeacher()
    write_jsonl(first, [response_for(r, teacher) for r in read_jsonl(root / 'teacher_requests.jsonl')])
    with pytest.raises(ValueError, match='20/30'):
        synth.finalize(root, responses_jsonl=first)
    assert not (root / 'data').exists()
    remaining = read_jsonl(root / 'verifier_requests.jsonl')
    assert len(remaining) == 10
    second = tmp_path / 'second.jsonl'
    write_jsonl(second, [response_for(r, teacher) for r in remaining])
    manifest = synth.finalize(root, responses_jsonl=second)
    assert manifest['synthesis']['budget']['api_attempt_records'] == 0
    assert manifest['synthesis']['budget']['import_attempt_records'] == 30
    assert synth.finalize(root, responses_jsonl=second) == manifest


def test_truncated_attempts_remain_evidence_and_budget_persists(tmp_path):
    root, _, _, cfg = make_run(tmp_path, max_calls=4)
    class Truncated(FakeTeacher):
        def chat(self, *a, **kw):
            result = super().chat(*a, **kw)
            result.raw['finish_reason'] = 'length'
            return result
    teacher = Truncated()
    status = synth.generate(root, cfg, teacher=teacher)
    assert not status['complete'] and status['accepted'] == 0 and len(teacher.calls) == 4
    items = read_jsonl(root / 'teacher_responses.jsonl')
    assert all(i['response']['text'] and i['attempt'] <= 1 for i in items)
    assert synth.generate(root, cfg, teacher=teacher, resume=True) == status
    assert len(teacher.calls) == 4
    with pytest.raises(ValueError, match='incomplete'):
        synth.finalize(root)


def test_unknown_metadata_does_not_become_zero_usage_or_pinned_model(tmp_path):
    root, _, _, cfg = make_run(tmp_path)
    class Unknown(FakeTeacher):
        def chat(self, *a, **kw):
            result = super().chat(*a, **kw)
            result.raw = {}
            return result
    status = synth.generate(root, cfg, teacher=Unknown())
    assert status['complete']
    assert status['budget']['unknown_usage_attempts'] == 30
    assert status['budget']['unknown_provider_request_attempt_records'] == 30
    assert status['budget']['observed_usage'] == {} and not status['budget']['accounting_complete']
    synth.finalize(root)
    row = read_jsonl(root / 'data' / 'extractor' / 'train.jsonl')[0]
    assert row['teacher_actual_model'] is None
    assert not row['quality_checks']['completion_metadata_known']
    assert all('prompt_tokens' not in args[1] for args, kwargs in DummyMonitor.usages)


def test_interrupted_call_consumes_budget_without_inventing_usage(tmp_path):
    root, _, _, cfg = make_run(tmp_path)
    request = read_jsonl(root / 'teacher_requests.jsonl')[0]
    synth._write_attempt(root, dict(schema_version=1, request_id=request['request_id'], attempt=0,
        origin='api', status='reserved', response=None, response_sha256=None,
        validation={'accepted': False, 'reasons': ['pending_call'], 'verdict': None}))
    teacher = FakeTeacher()
    status = synth.generate(root, cfg, teacher=teacher, resume=True)
    assert status['complete'] and len(teacher.calls) == 30
    assert status['budget']['attempts_consumed'] == 31
    assert status['budget']['unknown_outcome_attempts'] == 1
    assert not status['budget']['accounting_complete']


def test_changed_config_attempts_and_publication_refused(tmp_path):
    root, _, _, cfg = make_run(tmp_path)
    synth.generate(root, cfg, teacher=FakeTeacher())
    synth.finalize(root)
    with pytest.raises(ValueError, match='configuration changed'):
        synth.generate(root, {**cfg, 'seed': 43}, teacher=FakeTeacher(), resume=True)
    path = root / 'data' / 'reasoner' / 'train.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='publication changed'):
        synth.finalize(root)
    path = next((root / 'calls').glob('*/*.json'))
    item = json.loads(path.read_text())
    item['response']['text'] += ' changed'
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match='attempt changed'):
        synth.generate(root, cfg, teacher=FakeTeacher(), resume=True)


def test_teacher_transport_change_invalidates_resume(tmp_path, monkeypatch):
    root, _, _, cfg = make_run(tmp_path)
    hashes = synth._code_hashes()
    monkeypatch.setattr(synth, '_code_hashes', lambda: {**hashes, 'teachers/openai_client.py': 'changed'})
    with pytest.raises(ValueError, match='code/template changed'):
        synth.generate(root, cfg, teacher=FakeTeacher())


def test_import_identity_and_unknown_requests_refused(tmp_path):
    root, _, _, _ = make_run(tmp_path)
    path = tmp_path / 'bad.jsonl'
    request = read_jsonl(root / 'teacher_requests.jsonl')[0]
    item = response_for(request)
    item['request_id'] = 'unknown'
    write_jsonl(path, [item])
    with pytest.raises(ValueError, match='unknown'):
        synth.finalize(root, responses_jsonl=path)
    item = response_for(request)
    item['response']['provider'] = 'other'
    write_jsonl(path, [item])
    with pytest.raises(ValueError, match='provider/model'):
        synth.finalize(root, responses_jsonl=path)
    assert not list((root / 'calls').glob('*/*.json'))


def test_numerical_coincidence_is_not_misclassified_as_teacher_gt_leak(tmp_path):
    root, _, _, cfg = make_run(tmp_path)
    class Numeric(FakeTeacher):
        def chat(self, *a, **kw):
            result = super().chat(*a, **kw)
            if 'Extract givens' in a[0][0]['content']:
                result.text += ' The computed identifier is 99112233.'
            return result
    assert synth.generate(root, cfg, teacher=Numeric())['complete']


@pytest.mark.parametrize('text', ['FINAL_ANSWER: \\boxed{5}', '**FINAL_ANSWER:** \\boxed{5}'])
def test_candidates_need_derivation_not_only_answer(text):
    assert not synth._validate_response({'kind': 'candidate'}, {'text': text})['accepted']


@pytest.mark.parametrize('text', [
    'Verdict: correct\nEvidence: checked arithmetic\nCorrection: None needed',
    '**Verdict:** correct\n**Evidence:** checked arithmetic\n**Correction:** None needed',
    '__Verdict__: incorrect\n__Evidence__: 2+3=6 is false\n__Correction__: use 5',
    'Verdict: uncertain; Evidence: unproved lemma; Correction: establish that lemma',
])
def test_verifier_tolerates_runtime_compatible_markdown(text):
    assert synth._validate_response({'kind': 'verifier'}, {'text': text})['accepted']


@pytest.mark.parametrize('text', ['Verdict: correct',
    'Verdict: correct\nEvidence: checked\nVerdict: incorrect\nCorrection: fix',
    'Verdict: correct or incorrect\nEvidence: checked\nCorrection: None'])
def test_verifier_conflicting_or_missing_fields_rejected(text):
    assert not synth._validate_response({'kind': 'verifier'}, {'text': text})['accepted']


@pytest.mark.parametrize('key,value', [('latency_seconds', -1), ('latency_seconds', float('nan')),
    ('latency_seconds', 'fast'), ('request_attempts', 0), ('request_attempts', True),
    ('request_attempts', 'two'), ('provider_usage', 'unknown')])
def test_import_metadata_validated(key, value):
    request = {'provider': 'openai', 'model': 'gpt-4o'}
    with pytest.raises(ValueError):
        synth._normal_response(dict(text='help', provider='openai', model='gpt-4o', **{key: value}), request)


def test_durable_response_survives_telemetry_failure_no_duplicate_call(tmp_path, monkeypatch):
    root, _, _, cfg = make_run(tmp_path)
    class BrokenMonitor(DummyMonitor):
        def generation(self, *args, **kwargs):
            raise OSError('disk log write failed')
    monkeypatch.setattr(synth, 'Monitor', BrokenMonitor)
    teacher = FakeTeacher()
    with pytest.raises(OSError):
        synth.generate(root, cfg, teacher=teacher)
    attempts = synth._attempts(root)
    assert len(attempts) == 1 and attempts[0]['validation']['accepted']
    assert attempts[0]['response']['text']
    first = teacher.calls[0]
    monkeypatch.setattr(synth, 'Monitor', DummyMonitor)
    assert synth.generate(root, cfg, teacher=teacher, resume=True)['complete']
    assert len(teacher.calls) == 30
    assert teacher.calls.count(first) == 1


def test_pool_excludes_all_manager_splits_and_source_near_duplicates(tmp_path):
    from src.verifiable.data import identity
    manager = manager_data(tmp_path, [record(i)['problem'] for i in range(4)])
    rows = [record(i) for i in range(80)]
    # A near copy of a held-out Manager question and a near source duplicate.
    rows.extend([{**record(0), 'problem': record(0)['problem'] + ' Please explain.', 'problem_idx': 'heldout-near'},
                 {**record(4), 'problem': record(4)['problem'] + ' Please explain.', 'problem_idx': 'source-near'},
                 record(5)])
    raw = tmp_path / 'pool_raw.jsonl'
    write_jsonl(raw, rows)
    cfg = dict(train_size=3, dev_size=2, scan_limit=100, max_calls=60)
    root = tmp_path / 'synthesis'
    run = synth.prepare(root, manager, cfg, raw_jsonl=raw)
    assert run['stats']['manager_exact_overlap'] == 4
    assert run['stats']['manager_near_overlap'] == 1
    assert run['stats']['source_exact_duplicate'] == 1
    assert run['stats']['source_near_duplicate'] == 1
    pool = read_jsonl(root / 'pool.jsonl')
    excluded = {identity(record(i)['problem']) for i in range(4)}
    assert not excluded & {r['question_hash'] for r in pool}
    synth.generate(root, cfg, teacher=FakeTeacher())
    synth.finalize(root)
    membership = {}
    for role in KINDS:
        for split in ('train', 'dev'):
            for row in read_jsonl(root / 'data' / role / f'{split}.jsonl'):
                assert membership.setdefault(row['question_hash'], split) == split
    assert len(membership) == 5


def test_pool_selection_is_reproducible_and_source_mutation_blocks_resume(tmp_path):
    root, manager, raw, cfg = make_run(tmp_path)
    second = tmp_path / 'copy'
    run = synth.prepare(second, manager, cfg, raw_jsonl=raw)
    assert (second / 'synthesis_run.json').read_bytes() == (root / 'synthesis_run.json').read_bytes()
    assert (second / 'teacher_requests.jsonl').read_bytes() == (root / 'teacher_requests.jsonl').read_bytes()
    assert run['source']['upstream_revision_verified'] is False
    raw.write_text(raw.read_text() + '\n')
    with pytest.raises(ValueError, match='source or Manager exclusion changed'):
        synth.prepare(root, manager, cfg, raw_jsonl=raw, resume=True)
