"""GT-blind teacher synthesis for math experts, with export/import and resume.

prepare freezes a decontaminated question pool and exports extractor, reasoner,
and two independent solver requests per question. Candidate responses unlock
verifier requests. finalize publishes SFT data only after every frozen task has
an accepted teacher response. Numina answers/references are audit sidecars only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace
from urllib.parse import urlsplit

from . import expert_data as ed, protocol
from .answers import extract_final, equivalent
from .data import normalize, identity, verify_manifest
from .telemetry import Monitor


def _hash(value):
    return ed._digest(value)


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(ed._canonical(value) + '\n', encoding='utf-8')
    temporary.replace(path)


def _jsonl(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    ed._jsonl(temporary, rows)
    temporary.replace(path)


@contextmanager
def _lock(root):
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.synthesis.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another synthesis process is using this output') from exc
        yield


def load_config(config):
    raw = dict(config) if isinstance(config, dict) else json.loads(Path(config).read_text())
    allowed = {'provider', 'model', 'teacher_revision', 'base_url', 'temperature', 'candidate_temperature',
               'max_tokens', 'candidate_max_tokens', 'candidates_per_question', 'train_size', 'dev_size',
               'scan_limit', 'seed', 'max_retries', 'max_calls', 'source_revision', 'timeout', 'generation_surface'}
    if set(raw) - allowed:
        raise ValueError('Unknown synthesis config fields; credentials must never be stored in synthesis config')
    config = {**dict(provider='openai', model='gpt-4o', teacher_revision=None, base_url=None, generation_surface='api',
                    temperature=.2, candidate_temperature=.7, candidates_per_question=2,
                    train_size=128, dev_size=32, scan_limit=30000, seed=42, max_retries=1,
                    max_calls=1920, source_revision=ed.NUMINA_REVISION, timeout=120), **raw}
    defaults = dict(extractor=1024, reasoner=2048, candidate=4096, verifier=1536)
    if not isinstance(config.get('max_tokens', {}), dict):
        raise ValueError('max_tokens must be a per-role dictionary')
    if set(config.get('max_tokens', {})) - set(defaults):
        raise ValueError('Unknown max_tokens role')
    config['max_tokens'] = {**defaults, **config.get('max_tokens', {})}
    if 'candidate_max_tokens' in config:
        config['max_tokens']['candidate'] = config.pop('candidate_max_tokens')
    for key in ('train_size', 'dev_size', 'scan_limit', 'max_calls', 'timeout'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['seed']) is not int or type(config['max_retries']) is not int or config['max_retries'] < 0:
        raise ValueError('Invalid seed or retry limit')
    if config['candidates_per_question'] != 2:
        raise ValueError('This protocol requires two teacher candidate tasks per question')
    if config['generation_surface'] not in ('api', 'codex_subagent'):
        raise ValueError('generation_surface must be api or codex_subagent')
    if config['generation_surface'] == 'codex_subagent' and (
            config['provider'] != 'openai' or config['teacher_revision'] is not None or config['base_url'] is not None):
        raise ValueError('codex_subagent requires provider openai, unknown teacher_revision and no API base_url')
    for key in ('temperature', 'candidate_temperature'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or not 0 <= config[key] <= 2:
            raise ValueError(f'{key} must be finite and between zero and two')
    if any(type(n) is not int or n <= 0 for n in config['max_tokens'].values()):
        raise ValueError('Generation budgets must be positive integers')
    if not re.fullmatch(r'[0-9a-fA-F]{40}', config['source_revision']):
        raise ValueError('Numina requires a pinned commit revision')
    if any(not isinstance(config[k], str) or not config[k].strip() for k in ('provider', 'model')):
        raise ValueError('Teacher provider and model must be named explicitly')
    if config['teacher_revision'] is not None and not isinstance(config['teacher_revision'], str):
        raise ValueError('Teacher revision must be a string or explicit null for an API alias')
    if config['base_url'] is not None:
        url = urlsplit(config['base_url'])
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('base_url must not contain credentials, query strings or fragments')
    config['source_revision'] = config['source_revision'].lower()
    return config


def _teacher(config):
    return {'provider': config['provider'], 'model': config['model'], 'revision': config['teacher_revision']}


def _codex_context_claims():
    return {'task_messages_use_runtime_protocol': True, 'effective_prompt_identical_to_runtime': False,
            'gt_visible_to_teacher': None, 'no_gold_disclosed_scope': 'orchestrator_inputs_only',
            'evidence_scope': 'recorded orchestrator declarations; platform context and actual API model/sampling are not attested',
            'candidate_sampling': 'separate requested tasks with shared context within each batch; independent stochastic API sampling is not established'}


def _code_hashes():
    folder = Path(__file__).parent
    result = {name: ed._sha(folder / name) for name in
              ('expert_synthesis.py', 'expert_data.py', 'data.py', 'answers.py', 'protocol.py', 'chat_template.jinja')}
    result.update({'teachers/' + name: ed._sha(folder.parent / 'teachers' / name) for name in
                   ('base.py', 'openai_client.py', 'anthropic_client.py', 'deepseek_client.py', '_response_metadata.py')})
    return result


def _read_run(root, config=None):
    root = Path(root)
    run = json.loads((root / 'synthesis_run.json').read_text())
    payload = {k: v for k, v in run.items() if k != 'run_sha256'}
    if run.get('run_sha256') != _hash(payload):
        raise ValueError('Synthesis run fingerprint changed')
    if config is not None and run['config'] != load_config(config):
        raise ValueError('Synthesis configuration changed across resume')
    for filename, digest in run['sha256'].items():
        if filename not in {'pool.jsonl', 'references.jsonl', 'exclusions.jsonl'} or ed._sha(root / filename) != digest:
            raise ValueError(f'Frozen synthesis input changed: {filename}')
    current = _code_hashes()
    if run['code_sha256'] != current:
        raise ValueError('Synthesis code/template changed; use a new output to preserve provenance')
    return run, list(ed._local_rows(root / 'pool.jsonl'))


def _make_request(run, row, kind, variant=0, dependency=None):
    config = run['config']
    view = SimpleNamespace(question=row['question'], context=row.get('context', ''))
    candidate = dependency['response']['text'] if dependency else ''
    messages = protocol.messages(view, direct=True) if kind == 'candidate' else protocol.advisor_messages(kind, view, candidate)
    core = {'schema_version': 1, 'task_id': f"{row['question_hash']}:{kind}:{variant}", 'kind': kind,
            'variant': variant, 'question_hash': row['question_hash'], 'split': row['split'],
            'provider': config['provider'], 'model': config['model'], 'teacher_revision': config['teacher_revision'],
            'generation_surface': config['generation_surface'],
            'config_sha256': _hash(config), 'run_sha256': run['run_sha256'], 'messages': messages,
            'prompt_sha256': _hash(messages), 'temperature': config['candidate_temperature'] if kind == 'candidate' else config['temperature'],
            'max_tokens': config['max_tokens'][kind], 'candidate_request_id': dependency['request_id'] if dependency else None,
            'candidate_response_sha256': dependency['response_sha256'] if dependency else None}
    return {**core, 'request_id': _hash(core)}


def _attempts(root):
    items = []
    for path in sorted((Path(root) / 'calls').glob('*/*.json')):
        item = json.loads(path.read_text())
        payload = {k: v for k, v in item.items() if k != 'record_sha256'}
        if item.get('record_sha256') != _hash(payload):
            raise ValueError(f'Teacher attempt changed: {path.name}')
        if path.parent.name != item['request_id'] or path.stem != f"{item['attempt']:04d}":
            raise ValueError('Teacher attempt path identity mismatch')
        items.append(item)
    return items


def _accepted(items):
    accepted = {}
    for item in items:
        if item.get('validation', {}).get('accepted'):
            rid = item['request_id']
            if rid in accepted and accepted[rid]['response_sha256'] != item['response_sha256']:
                raise ValueError('Conflicting accepted teacher responses')
            if item['response_sha256'] != _hash(item['response']):
                raise ValueError('Teacher response fingerprint mismatch')
            accepted[rid] = item
    return accepted


def _requests(run, pool, accepted):
    requests = []
    for row in pool:
        requests.extend(_make_request(run, row, kind) for kind in ('extractor', 'reasoner'))
        for variant in range(2):
            candidate = _make_request(run, row, 'candidate', variant)
            requests.append(candidate)
            if candidate['request_id'] in accepted:
                requests.append(_make_request(run, row, 'verifier', variant, accepted[candidate['request_id']]))
    return requests


def _refresh(root, run, pool):
    items = _attempts(root)
    accepted = _accepted(items)
    requests = _requests(run, pool, accepted)
    by_id = {r['request_id']: r for r in requests}
    if any(item['request_id'] not in by_id for item in items):
        raise ValueError('Teacher attempt does not belong to this frozen synthesis DAG')
    for item in items:
        if item.get('response') is not None:
            request = by_id[item['request_id']]
            if _normal_response(item['response'], request) != item['response']:
                raise ValueError('Teacher response metadata changed')
            validation = _validate_response(request, item['response'])
            if item.get('validation') != validation or item.get('response_sha256') != _hash(item['response']):
                raise ValueError('Teacher attempt validation/content changed')
    _jsonl(Path(root) / 'teacher_requests.jsonl', requests)
    _jsonl(Path(root) / 'teacher_responses.jsonl', items)
    pending = [r for r in requests if r['request_id'] not in accepted]
    _jsonl(Path(root) / 'pending_requests.jsonl', pending)
    _jsonl(Path(root) / 'verifier_requests.jsonl', [r for r in pending if r['kind'] == 'verifier'])
    return requests, items, accepted


def _write_attempt(root, item):
    value = {**item, 'record_sha256': _hash(item)}
    _atomic(Path(root) / 'calls' / item['request_id'] / f"{item['attempt']:04d}.json", value)
    return value


def prepare(out_dir, manager_data_dir, config, raw_jsonl=None, resume=False):
    config = load_config(config)
    root, manager = Path(out_dir), Path(manager_data_dir)
    with _lock(root):
        manager_index, exclusion_source = ed._manager_snapshot(manager)
        source = {'dataset': ed.NUMINA_DATASET, 'revision': config['source_revision'], 'split': 'train',
                  'mode': 'local_raw_jsonl' if raw_jsonl else 'pinned_huggingface_parquet',
                  'upstream_revision_verified': raw_jsonl is None}
        if raw_jsonl:
            source['local_sha256'] = ed._sha(raw_jsonl)
        run_path = root / 'synthesis_run.json'
        if run_path.exists():
            if not resume:
                raise FileExistsError('Synthesis preparation exists; use --resume')
            run, pool = _read_run(root, config)
            if run['source'] != source or run['manager_exclusion'] != exclusion_source:
                raise ValueError('Synthesis source or Manager exclusion changed')
            _refresh(root, run, pool)
            return run
        if any(p.name != '.synthesis.lock' for p in root.iterdir()):
            raise FileExistsError('Incomplete/unrecognized preparation; choose a clean output')
        raw = ed._local_rows(raw_jsonl) if raw_jsonl else ed._remote_rows(config['source_revision'])
        eligible, exclusions, stats = {}, [], Counter()
        stream_sha = hashlib.sha256()
        try:
            for index, record in enumerate(itertools.islice(raw, config['scan_limit'])):
                stats['scanned'] += 1
                stream_sha.update((ed._canonical(record) + '\n').encode())
                row = normalize(record, 'numina', index)
                if row is None or ed._requires_image(row.question):
                    stats['source_quality_rejected'] += 1
                    continue
                key = identity(row.question)
                overlap = manager_index.match(row.question)
                if overlap:
                    reason = 'manager_' + overlap[1] + '_overlap'
                    stats[reason] += 1
                    exclusions.append({'question_hash': key, 'matched_hash': overlap[0], 'reason': reason})
                    continue
                stable_field = next((name for name in ('problem_idx', 'id', 'problem_id')
                                     if record.get(name) is not None and str(record[name]).strip()), None)
                common = {'question_hash': key, 'question': row.question, 'context': row.context,
                          'source_id': str(record[stable_field]) if stable_field else f'source-index:{index}',
                          'source_id_kind': stable_field or 'pinned_source_stream_index', 'source_ordinal': index,
                          'source_dataset': ed.NUMINA_DATASET, 'source_revision': config['source_revision'],
                          'source_record_sha256': _hash(record), 'source_collection': record.get('source'),
                          'reference_sha256': _hash(record.get('solution'))}
                item = {'common': common, 'ground_truth': row.ground_truth, 'solution': record.get('solution')}
                if key in eligible:
                    stats['source_exact_duplicate'] += 1
                    if common['source_record_sha256'] >= eligible[key]['common']['source_record_sha256']:
                        continue
                eligible[key] = item
        finally:
            if hasattr(raw, 'close'):
                raw.close()
        if raw_jsonl and ed._sha(raw_jsonl) != source['local_sha256']:
            raise ValueError('Raw source changed during preparation')
        verify_manifest(manager)
        if ed._sha(manager / 'manifest.json') != exclusion_source['manifest_sha256']:
            raise ValueError('Manager manifest changed during preparation')
        seen = ed._NearIndex()
        selected = {'train': [], 'dev': []}
        for key in sorted(eligible, key=lambda key: (_hash([config['seed'], key]), key)):
            item = eligible[key]
            overlap = seen.match(item['common']['question'])
            if overlap:
                stats['source_near_duplicate'] += 1
                exclusions.append({'question_hash': key, 'matched_hash': overlap[0], 'reason': 'source_near_duplicate'})
                continue
            seen.add(item['common']['question'], key)
            split = 'dev' if int(_hash(['expert-split-v1', config['seed'], key]), 16) % 5 == 0 else 'train'
            stats['eligible_' + split] += 1
            if len(selected[split]) < config[split + '_size']:
                selected[split].append({**item, 'common': {**item['common'], 'split': split}})
        if any(len(selected[split]) != config[split + '_size'] for split in selected):
            raise ValueError(f'Not enough disjoint teacher question groups: {dict(stats)}')
        pool = [item['common'] for split in selected for item in selected[split]]
        refs = [{**item['common'], 'ground_truth': item['ground_truth'], 'solution': item['solution'],
                 'reference_is_teacher_input': False, 'reviewed': False}
                for split in selected for item in selected[split]]
        files = {'pool.jsonl': pool, 'references.jsonl': refs, 'exclusions.jsonl': sorted(exclusions, key=ed._canonical)}
        for name, rows in files.items():
            _jsonl(root / name, rows)
        run = {'schema_version': 1, 'supervision': 'teacher_synthetic', 'config': config, 'config_sha256': _hash(config),
               'teacher': _teacher(config), 'source': source, 'scanned_records_sha256': stream_sha.hexdigest(),
               'manager_exclusion': exclusion_source, 'stats': dict(stats),
               'question_counts': {split: len(items) for split, items in selected.items()},
               'sha256': {name: ed._sha(root / name) for name in files},
               'code_sha256': _code_hashes(),
               'generation_surface': config['generation_surface'],
               'sampling_reproducibility': (
                   'seed fixes question selection and task identities; Codex subagent batches share context within each batch; '
                   'actual temperature, top_p and max_tokens are unknown; requested budgets are guidance only; '
                   'candidate tasks are not claimed to be independent stochastic API samples; exact responses are frozen for reuse'
                   if config['generation_surface'] == 'codex_subagent' else
                   'seed fixes question selection and task identities; TeacherClient.chat has no seed argument, so API generations are stochastic; exact responses are frozen for reuse'),
               'pool_rule': 'Numina quality + text-only question + Manager/test exclusion + exact/near dedup; no arithmetic/reference/extractor-rule gate',
               'split_rule': 'fixed question-hash 20% dev bucket, seed-hash order; all role/candidate variants stay in the same split',
               'expected_tasks': len(pool) * 6}
        if config['generation_surface'] == 'codex_subagent':
            run['execution_context'] = _codex_context_claims()
        run['run_sha256'] = _hash(run)
        _atomic(run_path, run)
        _refresh(root, run, pool)
    return run


def _codex_execution(value, request):
    """Validate scoped orchestrator declarations, not hidden infrastructure claims.

    Wrapper/output hashes bind externally archived exact UTF-8 text; only the
    input-request hash can be checked against the frozen request here.
    """
    fixed = {'schema_version': 1, 'surface': 'codex_subagent', 'selected_model': request['model'],
             'model_evidence_source': 'collaboration.spawn_agent', 'isolated_from_parent': True,
             'shared_batch_context': True, 'no_gold_disclosed': True,
             'no_gold_disclosed_scope': 'orchestrator_inputs_only',
             'tool_policy': 'file_io_only_no_external_lookup_or_computation', 'reasoning_effort': None,
             'tools_used': None, 'tool_use_evidence_source': 'not_independently_observed',
             'actual_sampling': {'temperature': None, 'top_p': None, 'max_tokens': None},
             'requested_budgets_scope': 'guidance_only'}
    names = set(fixed) | {'agent_id', 'task_path', 'batch_id', 'input_request_sha256',
                         'batch_input_sha256', 'wrapper_sha256', 'raw_output_sha256'}
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError('Codex response requires complete execution provenance with no unknown fields')
    if any(value[key] != expected or type(value[key]) is not type(expected) for key, expected in fixed.items()):
        raise ValueError('Codex execution provenance or scoped declaration differs from the required protocol')
    for key in ('agent_id', 'task_path', 'batch_id'):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f'Codex execution requires a nonempty {key}')
    for key in ('input_request_sha256', 'batch_input_sha256', 'wrapper_sha256', 'raw_output_sha256'):
        if not isinstance(value[key], str) or not re.fullmatch(r'[0-9a-f]{64}', value[key]):
            raise ValueError(f'Codex execution requires a SHA256 {key}')
    if value['input_request_sha256'] != _hash(request):
        raise ValueError('Codex execution input request fingerprint differs from the frozen request')
    return value


def _normal_response(value, request):
    if not isinstance(value, dict) or not isinstance(value.get('text'), str):
        raise ValueError('Imported response must contain original teacher text')
    # No opaque SDK response/header/key objects are persisted. Preserve selected
    # provenance/usage fields, with null for missing provider evidence.
    response = {key: value.get(key) for key in ('text', 'provider', 'model', 'actual_model', 'usage',
        'finish_reason', 'request_id', 'system_fingerprint', 'latency_seconds', 'request_attempts', 'provider_usage')}
    if response['provider'] != request['provider'] or response['model'] != request['model']:
        raise ValueError('Teacher response provider/model does not match the frozen request')
    for key in ('actual_model', 'finish_reason', 'request_id', 'system_fingerprint'):
        if response[key] is not None and not isinstance(response[key], str):
            raise ValueError(f'Invalid teacher {key}')
    if response['usage'] is not None:
        if not isinstance(response['usage'], dict):
            raise ValueError('Teacher usage must be an object or null')
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            val = response['usage'].get(key)
            if val is not None and (type(val) is not int or val < 0):
                raise ValueError('Token usage must be a nonnegative integer or unknown')
    if response['latency_seconds'] is not None and (type(response['latency_seconds']) not in (int, float)
            or not math.isfinite(response['latency_seconds']) or response['latency_seconds'] < 0):
        raise ValueError('Teacher latency_seconds must be finite and nonnegative or null')
    if response['request_attempts'] is not None and (type(response['request_attempts']) is not int or response['request_attempts'] < 1):
        raise ValueError('Teacher request_attempts must be a positive integer or null')
    if response['provider_usage'] is not None and not isinstance(response['provider_usage'], dict):
        raise ValueError('Teacher provider_usage must be an object or null')
    if request.get('generation_surface', 'api') == 'codex_subagent':
        response['execution'] = _codex_execution(value.get('execution'), request)
        if any(response[key] is not None for key in ('actual_model', 'usage', 'finish_reason',
                'request_id', 'system_fingerprint', 'request_attempts', 'provider_usage')):
            raise ValueError('Codex subagent responses require unknown API model, completion and usage metadata to remain null')
    return response


def _validate_response(request, response):
    reasons, verdict = [], None
    text = response['text']
    if response.get('finish_reason') in {'length', 'max_tokens', 'content_filter'}:
        reasons.append('incomplete_or_filtered_teacher_response')
    if not text.strip():
        reasons.append('empty_teacher_response')
    if '<tool_call' in text:
        reasons.append('unexpected_tool_call')
    if request['kind'] == 'candidate':
        declaration = re.search(r'(?im)^\s*(?:\*\*|__)?FINAL_ANSWER\s*:', text)
        if extract_final(text) is None or declaration is None or not text[:declaration.start()].strip():
            reasons.append('candidate_requires_derivation_and_terminal_answer')
    if request['kind'] == 'verifier':
        # The runtime asks for three fields, not one particular Markdown layout.
        # Accept emphasis and inline semicolon separators without changing text.
        labels = list(re.finditer(r'(?im)(?:^|[;\n])\s*(?:[-*]\s+)?(?:\*\*|__)?(Verdict|Evidence|Correction)(?:\*\*|__)?\s*:\s*(?:\*\*|__)?', text))
        values = [text[m.end():labels[i + 1].start() if i + 1 < len(labels) else len(text)].strip()
                  for i, m in enumerate(labels)]
        named = [m[1].lower() for m in labels]
        final_verdict = values[0].strip(' *_!.').lower() if values else ''
        if (named == ['verdict', 'evidence', 'correction'] and not text[:labels[0].start()].strip()
                and all(values) and final_verdict in {'correct', 'incorrect', 'uncertain'}):
            verdict = final_verdict
        else:
            reasons.append('verifier_requires_verdict_evidence_correction')
    return {'accepted': not reasons, 'reasons': reasons, 'verdict': verdict}


def _commit_response(root, request, value, origin, run, reserved=None):
    response = _normal_response(value, request)
    items = _attempts(root)
    accepted = _accepted(items)
    digest = _hash(response)
    if request['request_id'] in accepted:
        if accepted[request['request_id']]['response_sha256'] != digest:
            raise ValueError('Cannot replace an already accepted teacher response')
        return accepted[request['request_id']]
    prior = [item for item in items if item['request_id'] == request['request_id']]
    if reserved is None and any(item.get('response_sha256') == digest for item in prior):
        return next(item for item in prior if item.get('response_sha256') == digest)
    attempt = reserved['attempt'] if reserved else len(prior)
    if reserved is None and (len(items) >= run['config']['max_calls'] or attempt > run['config']['max_retries']):
        raise ValueError('Persistent teacher call/retry budget exhausted')
    record = {'schema_version': 1, 'request_id': request['request_id'], 'attempt': attempt, 'origin': origin,
              'status': 'completed', 'response': response, 'response_sha256': digest,
              'validation': _validate_response(request, response)}
    return _write_attempt(root, record)


def _import(root, run, pool, path):
    # Process ordinary batches in any order: a verifier response is accepted
    # only after its candidate response has generated that exact request ID.
    pending = list(ed._local_rows(path))
    while pending:
        requests, _, _ = _refresh(root, run, pool)
        by_id = {r['request_id']: r for r in requests}
        deferred, changed = [], False
        for row in pending:
            if row.get('request_id') not in by_id:
                deferred.append(row)
                continue
            value = row.get('response') if isinstance(row.get('response'), dict) else {k: v for k, v in row.items() if k != 'request_id'}
            _commit_response(root, by_id[row['request_id']], value, 'external_import', run)
            changed = True
        if deferred and not changed:
            raise ValueError('Imported responses contain unknown, changed, or dependency-ineligible request IDs')
        pending = deferred
    _refresh(root, run, pool)


def generate(out_dir, config, teacher=None, resume=False):
    """Execute a bounded DAG. Inject a fake TeacherClient for offline tests.

    CLI construction of an API client occurs only in this explicit subcommand;
    prepare/finalize/export/import never instantiate a client or read secrets.
    """
    root = Path(out_dir)
    with _lock(root):
        run, pool = _read_run(root, config)
        if run['config']['generation_surface'] == 'codex_subagent':
            raise ValueError('codex_subagent is external-only; import recorded responses with finalize --responses-jsonl')
        if _attempts(root) and not resume:
            raise FileExistsError('Teacher attempts exist; use --resume')
        with Monitor(root, 'expert_teacher_synthesis') as monitor:
            if teacher is None:
                from ..teachers.base import build_teacher_client
                teacher = build_teacher_client(run['config']['provider'], run['config']['model'],
                    timeout=run['config']['timeout'], max_retries=0, base_url=run['config']['base_url'])
            if teacher.provider != run['config']['provider'] or teacher.model != run['config']['model']:
                raise ValueError('Injected teacher identity differs from synthesis configuration')
            while True:
                requests, items, accepted = _refresh(root, run, pool)
                # An unclean exit may have incurred a charge. Keep its reservation,
                # count it against the limit, and spend at most the explicit retry.
                for item in items:
                    if item['status'] == 'reserved':
                        replacement = {k: v for k, v in item.items() if k != 'record_sha256'}
                        replacement.update(status='interrupted_unknown', validation={'accepted': False,
                            'reasons': ['interrupted_call_outcome_unknown'], 'verdict': None})
                        _write_attempt(root, replacement)
                items = _attempts(root)
                counts = Counter(item['request_id'] for item in items)
                pending = [r for r in requests if r['request_id'] not in accepted and counts[r['request_id']] <= run['config']['max_retries']]
                if not pending or len(items) >= run['config']['max_calls']:
                    break
                request = pending[0]
                reservation = {'schema_version': 1, 'request_id': request['request_id'], 'attempt': counts[request['request_id']],
                               'origin': 'api', 'status': 'reserved', 'response': None, 'response_sha256': None,
                               'validation': {'accepted': False, 'reasons': ['pending_call'], 'verdict': None}}
                monitor.update(phase=request['kind'], request_id=request['request_id'], question_hash=request['question_hash'],
                               split=request['split'], accepted=len(accepted), expected=run['expected_tasks'], attempts_consumed=len(items)+1)
                _write_attempt(root, reservation)
                try:
                    result = teacher.chat(request['messages'], temperature=request['temperature'], max_tokens=request['max_tokens'])
                    raw = result.raw or {}
                    value = {'text': result.text, 'provider': result.provider, 'model': result.model,
                             'actual_model': raw.get('model'), 'usage': raw.get('usage'), 'finish_reason': raw.get('finish_reason'),
                             'request_id': raw.get('id'), 'system_fingerprint': raw.get('system_fingerprint'),
                             'latency_seconds': raw.get('latency_seconds'), 'request_attempts': raw.get('request_attempts'),
                             'provider_usage': raw.get('provider_usage')}
                    result_record = _commit_response(root, request, value, 'api', run, reserved=reservation)
                except Exception as exc:
                    # Avoid persisting arbitrary exception text which can contain
                    # request URLs or credential-bearing provider error details.
                    reservation.update(status='error', validation={'accepted': False,
                        'reasons': ['teacher_error:' + type(exc).__name__], 'verdict': None})
                    _write_attempt(root, reservation)
                    monitor.event('teacher_call_failed', request_id=request['request_id'], phase=request['kind'], error_type=type(exc).__name__)
                    continue
                # Durable responses survive telemetry failures. Do not overwrite
                # a paid, accepted response with an unrelated logging exception.
                _log_response(monitor, request, result_record)
            requests, items, accepted = _refresh(root, run, pool)
            status = _status(run, requests, items, accepted)
            _atomic(root / 'synthesis_status.json', status)
            monitor.summary({'synthesis': status, 'synthesis_complete': status['complete'], 'supervision': 'teacher_synthetic'})
            return status


def _log_response(monitor, request, item):
    response = item['response']
    tokens = response['usage'] or {}
    observed = {key: tokens[key] for key in ('prompt_tokens', 'completion_tokens') if tokens.get(key) is not None}
    if response['latency_seconds'] is not None:
        observed['seconds'] = response['latency_seconds']
    unknown = any(tokens.get(key) is None for key in ('prompt_tokens', 'completion_tokens'))
    if unknown:
        monitor.state['usage_incomplete'] = True
    monitor.usage('teacher', {**observed, 'finish_reason': response['finish_reason'],
                  'valid_output': item['validation']['accepted']}, phase=request['kind'], request_id=request['request_id'],
                  usage_incomplete=unknown, provider=request['provider'], model=request['model'])
    monitor.event('teacher_response', request_id=request['request_id'], phase=request['kind'],
                  accepted=item['validation']['accepted'], reasons=item['validation']['reasons'],
                  usage_known=not unknown, response_sha256=item['response_sha256'])
    monitor.generation('teacher', {**observed, 'text': response['text'],
                       'truncated': response['finish_reason'] in {'length', 'max_tokens'},
                       'finish_reason': response['finish_reason'], 'valid_output': item['validation']['accepted']},
                       messages=request['messages'])


def _status(run, requests, items, accepted):
    usage = Counter()
    unknown_usage = 0
    provider_attempts, unknown_provider_attempts = 0, 0
    for item in items:
        response = item.get('response') or {}
        tokens = response.get('usage')
        if response.get('request_attempts') is None:
            unknown_provider_attempts += 1
        else:
            provider_attempts += response['request_attempts']
        if not isinstance(tokens, dict) or any(tokens.get(k) is None for k in ('prompt_tokens', 'completion_tokens')):
            unknown_usage += 1
        if isinstance(tokens, dict):
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if type(tokens.get(key)) is int:
                    usage[key] += tokens[key]
    rejected = Counter(reason for item in items for reason in item.get('validation', {}).get('reasons', []))
    return {'expected_tasks': run['expected_tasks'], 'available_requests': len(requests), 'accepted': len(accepted),
            'rejected': dict(rejected), 'complete': len(accepted) == run['expected_tasks'],
            'budget': {'max_calls': run['config']['max_calls'], 'attempts_consumed': len(items),
                       'api_attempt_records': sum(item['origin'] == 'api' for item in items),
                       'import_attempt_records': sum(item['origin'] == 'external_import' for item in items),
                       'unknown_outcome_attempts': sum(item['status'] in {'reserved', 'interrupted_unknown', 'error'} for item in items),
                       'observed_usage': dict(usage), 'unknown_usage_attempts': unknown_usage,
                       'observed_provider_request_attempts': provider_attempts,
                       'unknown_provider_request_attempt_records': unknown_provider_attempts,
                       'accounting_complete': unknown_usage == 0,
                       'usage_scope': 'recorded teacher response usage, including imported evidence; not a billing statement for this process',
                       'meaning': 'bounded teacher task attempts including failures/retries; unknown provider charges are not zero; imports do not call an API'}}


def finalize(out_dir, data_out=None, responses_jsonl=None):
    root = Path(out_dir)
    target = Path(data_out) if data_out else root / 'data'
    with _lock(root):
        run, pool = _read_run(root)
        if responses_jsonl:
            _import(root, run, pool, responses_jsonl)
        requests, items, accepted = _refresh(root, run, pool)
        status = _status(run, requests, items, accepted)
        _atomic(root / 'synthesis_status.json', status)
        if not status['complete']:
            raise ValueError(f"Teacher synthesis incomplete: {len(accepted)}/{run['expected_tasks']} accepted. Review pending_requests.jsonl and verifier_requests.jsonl; no SFT bundle published")
        refs = {row['question_hash']: row for row in ed._local_rows(root / 'references.jsonl')}
        codex = run['config']['generation_surface'] == 'codex_subagent'
        by_question = {row['question_hash']: row for row in pool}
        outputs = {f'{role}/{split}.jsonl': [] for role in protocol.KINDS for split in ('train', 'dev')}
        for request in requests:
            if request['kind'] == 'candidate':
                continue
            row = by_question[request['question_hash']]
            evidence = accepted[request['request_id']]
            response = evidence['response']
            candidate_evidence = accepted.get(request['candidate_request_id'])
            candidate = candidate_evidence['response']['text'] if candidate_evidence else ''
            entry = {**row, 'schema_version': 1, 'role': request['kind'], 'candidate': candidate,
                     'prompt': request['messages'], 'response': response['text'], 'supervision': 'teacher_synthetic',
                     'label_source': 'teacher_synthetic', 'reviewed': False,
                     'teacher_provider': run['config']['provider'], 'teacher_model': run['config']['model'],
                     'teacher_actual_model': response['actual_model'], 'teacher_revision': run['config']['teacher_revision'],
                     'teacher_request_id': request['request_id'], 'teacher_prompt_sha256': request['prompt_sha256'],
                     'teacher_response_sha256': evidence['response_sha256'], 'response_sha256': hashlib.sha256(response['text'].encode()).hexdigest(),
                     'template_sha256': run['code_sha256']['chat_template.jinja'], 'protocol_sha256': run['code_sha256']['protocol.py'],
                     'quality_checks': {'gt_visible_to_teacher': False, 'runtime_prompt_aligned': True,
                                        'label_verification': 'unverified_teacher_judgment',
                                        'completion_metadata_known': response['finish_reason'] is not None}}
            if codex:
                entry['teacher_generation_surface'] = 'codex_subagent'
                entry['teacher_execution'] = response['execution']
                entry['quality_checks'].update(_codex_context_claims(), runtime_prompt_aligned=False)
            if candidate_evidence:
                final = extract_final(candidate)
                entry.update(verdict=evidence['validation']['verdict'],
                    candidate_source='teacher_codex_subagent_solution' if codex else 'teacher_independent_solution',
                    candidate_request_id=request['candidate_request_id'], candidate_response_sha256=request['candidate_response_sha256'],
                    candidate_hash=hashlib.sha256(candidate.encode()).hexdigest(),
                    candidate_terminal_diagnostic={'extracted_answer': final,
                        'terminal_correct': equivalent(final, refs[row['question_hash']]['ground_truth']),
                        'scope': 'terminal_answer_only_not_full_derivation_label', 'used_for_teacher_verdict': False})
            outputs[f"{entry['role']}/{entry['split']}.jsonl"].append(entry)
        outputs.update({name: list(ed._local_rows(root / name)) for name in
                        ('references.jsonl', 'exclusions.jsonl', 'teacher_requests.jsonl', 'teacher_responses.jsonl')})
        counts = {role: {split: len(outputs[f'{role}/{split}.jsonl']) for split in ('train', 'dev')} for role in protocol.KINDS}
        candidates = {split: {row['candidate_hash'] for row in outputs[f'verifier/{split}.jsonl']} for split in ('train', 'dev')}
        manifest = {'schema_version': 1, 'supervision': 'teacher_synthetic', 'teacher': run['teacher'],
                    'generation_surface': run['config']['generation_surface'],
                    'synthesis': {'config': run['config'], 'config_sha256': run['config_sha256'], 'run_sha256': run['run_sha256'], **status},
                    'sampling_reproducibility': run['sampling_reproducibility'],
                    'source': run['source'], 'manager_exclusion': run['manager_exclusion'], 'counts': counts,
                    'question_counts': run['question_counts'], 'template_sha256': run['code_sha256']['chat_template.jinja'],
                    'protocol_sha256': run['code_sha256']['protocol.py'], 'split_rule': run['split_rule'], 'pool_rule': run['pool_rule'],
                    'quality_audit': {'teacher_used': True, 'reviewed': False, 'reviewed_examples': 0,
                                      'gt_visible_to_teacher': False, 'verifier_labels': 'unverified_teacher_judgments_on_complete_candidate_derivations',
                                      'candidate_terminal_diagnostic_is_verdict_label': False,
                                      'verifier_verdict_counts': dict(Counter(row['verdict'] for split in ('train', 'dev') for row in outputs[f'verifier/{split}.jsonl'])),
                                      'verifier_candidate_audit': {'unique_counts': {**{s: len(v) for s, v in candidates.items()}, 'all': len(candidates['train'] | candidates['dev'])},
                                          'cross_split_overlap_count': len(candidates['train'] & candidates['dev'])},
                                      'independent_benchmark': False, 'semantic_decontamination_proven': False}}
        if codex:
            manifest['execution_context'] = _codex_context_claims()
            manifest['quality_audit'].update(_codex_context_claims())
        # Hash the exact would-be publication before touching an existing bundle.
        manifest['sha256'] = {name: hashlib.sha256(''.join(ed._canonical(row) + '\n' for row in rows).encode()).hexdigest() for name, rows in outputs.items()}
        manifest['manifest_content_sha256'] = _hash(manifest)
        if target.exists() and any(target.iterdir()):
            existing = json.loads((target / 'manifest.json').read_text())
            if existing != manifest or any(ed._sha(target / name) != digest for name, digest in manifest['sha256'].items()):
                raise ValueError('Existing SFT publication changed or differs from the completed synthesis')
            return manifest
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.' + target.name + '.publishing-', dir=target.parent))
        try:
            for name, rows in outputs.items():
                ed._jsonl(temporary / name, rows)
            _atomic(temporary / 'manifest.json', manifest)
            if target.exists():
                target.rmdir()
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'generate', 'finalize'):
        sub = commands.add_parser(command)
        sub.add_argument('--out', required=True)
        if command in ('prepare', 'generate'):
            sub.add_argument('--config', required=True)
            sub.add_argument('--resume', action='store_true')
        if command == 'prepare':
            sub.add_argument('--manager-data-dir', required=True)
            sub.add_argument('--raw-jsonl')
        if command == 'finalize':
            sub.add_argument('--data-out')
            sub.add_argument('--responses-jsonl')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.out, args.manager_data_dir, args.config, raw_jsonl=args.raw_jsonl, resume=args.resume)
        summary = {'prepared': True, 'expected_tasks': result['expected_tasks'], 'question_counts': result['question_counts']}
    elif args.command == 'generate':
        summary = generate(args.out, args.config, resume=args.resume)
    else:
        result = finalize(args.out, args.data_out, args.responses_jsonl)
        summary = {'published': True, 'counts': result['counts'], 'supervision': result['supervision']}
    print(json.dumps(summary, ensure_ascii=False))
    if args.command == 'generate' and not summary['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
