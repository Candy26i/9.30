"""Explicit, append-only exception to one frozen Codex import retry limit.

This adapter never calls a model and never changes teacher text, original run,
configuration, request IDs, or the original validator. A frozen sidecar permits
one additional candidate-format attempt for named requests only. The original
publication is kept separately; only the derived publication advertises the
amended admission policy. It is not an amendment to model prompts or grading.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

PROTOCOL = 'codex-budget-continuation-v1'
REASON = 'candidate_requires_derivation_and_terminal_answer'
ORIGIN_PREFIX = 'external_import_continuation:'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def immutable_json(path, value):
    """Never overwrite an established sidecar, including on repeated freeze."""
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        require(path.read_text() == data, f'Frozen continuation artifact changed: {path.name}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(data)


def module(package):
    sys.path.insert(0, str(Path(package).resolve()))
    from src.verifiable import expert_synthesis
    return expert_synthesis


def ledger_snapshot(synthesis):
    return {str(path.relative_to(synthesis)): digest(path)
            for path in sorted((synthesis / 'calls').glob('*/*.json'))}


def assignments(root):
    """Count every successful dispatch assignment, including no-output/ID failures.

    A row assigned in two fresh batches counts twice even when an earlier batch
    wrote no result or its ID could not be imported. Unknown usage stays unknown.
    """
    result = {}
    for path in sorted((root / 'codex_dispatches').glob('*.json')):
        dispatch = read(path)
        bid = dispatch['batch_id']
        require(path.stem == bid and Path(bid).name == bid, 'Unsafe dispatch identity')
        job = read(root / 'codex_jobs' / f'{bid}.json')
        inp = root / 'codex_inputs' / f'{bid}.json'
        wrapper = root / 'codex_wrappers' / f'{bid}.txt'
        batch = read(inp)
        require(isinstance(batch, list) and len(batch) == job['count'] > 0, 'Assignment count differs')
        require(job['batch_id'] == bid, 'Assignment job identity differs')
        require(digest(inp) == job['batch_input_sha256'], 'Dispatched input changed')
        require(digest(wrapper) == job['wrapper_sha256'] == dispatch['wrapper_sha256'], 'Dispatched wrapper changed')
        require(dispatch['model'] == job['selected_model'] == 'gpt-6-luna'
                and dispatch['fork_turns'] == job['fork_turns'] == 'none', 'Dispatch model/context differs')
        if 'instructions_sha256' in job or 'instructions_sha256' in dispatch:
            require(digest(root / 'codex_instructions' / f'{bid}.txt')
                    == job.get('instructions_sha256') == dispatch.get('instructions_sha256'),
                    'Dispatched full instructions changed')
        result[bid] = {'assigned_requests': len(batch), 'dispatch_sha256': digest(path),
                       'request_ids': [r['request_id'] for r in batch]}
    return result


def state(root, synth):
    synthesis = root / 'synthesis'
    # This still checks the original source/template/input fingerprints exactly.
    run, pool = synth._read_run(synthesis)
    require(run['config']['generation_surface'] == 'codex_subagent'
            and run['config']['model'] == 'gpt-6-luna', 'Continuation only supports this Codex Luna protocol')
    require(run['config']['max_retries'] == 1 and run['config']['max_calls'] == 1920,
            'Original retry/global policy differs')
    items = synth._attempts(synthesis)
    requests = {r['request_id']: r for r in synth._requests(run, pool, synth._accepted(items))}
    for item in items:
        require(item['request_id'] in requests, 'Attempt outside frozen request DAG')
        request = requests[item['request_id']]
        require(item.get('response') is not None, 'Unexpected missing Codex response')
        response = synth._normal_response(item['response'], request)
        require(response == item['response'] and synth._hash(response) == item['response_sha256'], 'Response changed')
        require(synth._validate_response(request, response) == item['validation'], 'Original validation changed')
    assigned = assignments(root)
    total = sum(b['assigned_requests'] for b in assigned.values())
    require(total <= 1920 and len(items) <= 1920, 'Global 1920 assignment/import budget exceeded')
    require(total >= len(items), 'Imported attempts exceed recorded dispatched assignments')
    return run, pool, items, requests, assigned


def exhausted(request, prior):
    require(request['kind'] == 'candidate', 'Only candidate-format exhaustion qualifies')
    require(len(prior) == 2 and sorted(i['attempt'] for i in prior) == [0, 1], 'Request has not exhausted exactly two attempts')
    require(all(i['status'] == 'completed' and i['validation'] ==
                {'accepted': False, 'reasons': [REASON], 'verdict': None} for i in prior),
            'Prior failures must solely be candidate terminal-format failures')


def freeze(root, package, amendment_id, request_ids):
    root = Path(root).resolve()
    require(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', amendment_id), 'Unsafe amendment ID')
    require(request_ids and len(set(request_ids)) == len(request_ids), 'Supply distinct exhausted request IDs')
    synth = module(package)
    with synth._lock(root / 'synthesis'):
        run, _, items, requests, assigned = state(root, synth)
        overrides = {}
        for rid in sorted(request_ids):
            require(rid in requests, 'Unknown frozen request')
            prior = sorted((i for i in items if i['request_id'] == rid), key=lambda i: i['attempt'])
            exhausted(requests[rid], prior)
            overrides[rid] = {'task_id': requests[rid]['task_id'], 'kind': 'candidate',
                              'original_max_retries': 1, 'effective_max_retries': 2,
                              'permitted_attempt': 2, 'prior_record_sha256': [i['record_sha256'] for i in prior]}
        # Multiple overlapping amendments would obscure the single extra attempt.
        for path in (root / 'codex_continuations').glob('*.json'):
            old = read(path)
            if old.get('protocol') == PROTOCOL:
                require(not set(overrides).intersection(old.get('request_overrides', {})), 'Request already has a frozen amendment')
        synthesis = root / 'synthesis'
        value = {'schema_version': 1, 'protocol': PROTOCOL, 'amendment_id': amendment_id,
                 'selection_basis': 'candidate_format_exhaustion_only_no_reference_or_performance_selection',
                 'original_run_sha256': run['run_sha256'], 'original_run_file_sha256': digest(synthesis / 'synthesis_run.json'),
                 'original_config_sha256': run['config_sha256'], 'original_config': run['config'],
                 'original_code_sha256': run['code_sha256'], 'adapter_sha256': digest(Path(__file__)),
                 'request_namespace': 'unchanged_original_frozen_run', 'original_max_retries': 1,
                 'global_max_calls': 1920, 'request_overrides': overrides,
                 'pre_amendment_ledger': ledger_snapshot(synthesis),
                 'pre_amendment_dispatches': {k: v['dispatch_sha256'] for k, v in assigned.items()},
                 'pre_amendment_assignment_count': sum(v['assigned_requests'] for v in assigned.values()),
                 'teacher_text_modified': False, 'validator_modified': False,
                 'usage_scope': 'counted request assignments; actual mathematical attempts and model usage may be unknown'}
        value['amendment_sha256'] = synth._hash(value)
        path = root / 'codex_continuations' / f'{amendment_id}.json'
        immutable_json(path, value)
    return {'amendment_path': str(path), 'amendment_sha256': value['amendment_sha256'],
            'request_ids': sorted(overrides)}


def verify_amendment(root, synth, path, current=None):
    amendment = read(path)
    require(amendment.get('protocol') == PROTOCOL and amendment.get('schema_version') == 1, 'Unknown continuation protocol')
    require(amendment.get('amendment_sha256') == synth._hash({k: v for k, v in amendment.items() if k != 'amendment_sha256'}),
            'Continuation fingerprint changed')
    require(amendment['adapter_sha256'] == digest(Path(__file__)), 'Continuation adapter source changed')
    run, pool, items, requests, assigned = current or state(root, synth)
    require(amendment['original_run_file_sha256'] == digest(root / 'synthesis/synthesis_run.json')
            and amendment['original_run_sha256'] == run['run_sha256']
            and amendment['original_config_sha256'] == run['config_sha256']
            and amendment['original_config'] == run['config']
            and amendment['original_code_sha256'] == run['code_sha256'], 'Original run/config/source differs from amendment')
    require(amendment['global_max_calls'] == 1920 and amendment['original_max_retries'] == 1
            and amendment['teacher_text_modified'] is False and amendment['validator_modified'] is False,
            'Continuation changes forbidden policy')
    require(amendment['request_namespace'] == 'unchanged_original_frozen_run'
            and amendment['selection_basis'] == 'candidate_format_exhaustion_only_no_reference_or_performance_selection',
            'Continuation selection or namespace changed')
    require(isinstance(amendment['request_overrides'], dict) and amendment['request_overrides'], 'Empty continuation whitelist')
    for relative, expected in amendment['pre_amendment_ledger'].items():
        require(re.fullmatch(r'calls/[0-9a-f]{64}/[0-9]{4}\.json', relative), 'Unsafe ledger snapshot path')
        require(digest(root / 'synthesis' / relative) == expected, 'Pre-amendment ledger changed')
    for bid, expected in amendment['pre_amendment_dispatches'].items():
        require(bid in assigned and assigned[bid]['dispatch_sha256'] == expected, 'Pre-amendment dispatch changed')
    require(amendment['pre_amendment_assignment_count'] == sum(assigned[bid]['assigned_requests']
            for bid in amendment['pre_amendment_dispatches']), 'Snapshot assignment count differs')
    for rid, override in amendment['request_overrides'].items():
        require(rid in requests and override == {'task_id': requests[rid]['task_id'], 'kind': 'candidate',
                'original_max_retries': 1, 'effective_max_retries': 2, 'permitted_attempt': 2,
                'prior_record_sha256': override.get('prior_record_sha256')}, 'Invalid request-specific continuation cap')
        prior = sorted((i for i in items if i['request_id'] == rid and i['attempt'] < 2), key=lambda i: i['attempt'])
        exhausted(requests[rid], prior)
        require([i['record_sha256'] for i in prior] == override['prior_record_sha256'], 'Prior failed records differ')
        require(all(f"calls/{rid}/{i['attempt']:04d}.json" in amendment['pre_amendment_ledger'] for i in prior),
                'Exhausted records not present in frozen snapshot')
    return amendment, (run, pool, items, requests, assigned)


def bind_import(root, synth, path, requests):
    """Require exact new batch IDs/text; no repair policy is introduced here."""
    path = Path(path).resolve()
    require(path.parent == (root / 'codex_imports').resolve() and path.suffix == '.jsonl', 'Use a sealed local Codex import')
    bid = path.stem
    receipt = read(root / 'codex_receipts' / f'{bid}.json')
    job = read(root / 'codex_jobs' / f'{bid}.json')
    dispatch = read(root / 'codex_dispatches' / f'{bid}.json')
    inp, raw = root / 'codex_inputs' / f'{bid}.json', root / 'codex_outputs' / f'{bid}.json'
    batch, results, imports = read(inp), read(raw), rows(path)
    require(digest(path) == receipt['import_sha256'], 'Sealed import changed')
    require(digest(raw) == receipt['raw_output_sha256'], 'Raw teacher output changed')
    require(digest(inp) == job['batch_input_sha256'] == receipt['batch_input_sha256'], 'Batch input changed')
    wrapper_hash = digest(root / 'codex_wrappers' / f'{bid}.txt')
    require(wrapper_hash == job['wrapper_sha256'] == receipt['wrapper_sha256'] == dispatch['wrapper_sha256'], 'Wrapper binding differs')
    require(dispatch['task_path'] == receipt['task_path'], 'Agent binding differs')
    if any('instructions_sha256' in v for v in (job, receipt, dispatch)):
        require(digest(root / 'codex_instructions' / f'{bid}.txt') == job.get('instructions_sha256')
                == receipt.get('instructions_sha256') == dispatch.get('instructions_sha256'),
                'Full generation instructions changed')
    require(len(imports) == len(batch) == len(results) == job['count']
            and [x['request_id'] for x in imports] == [x['request_id'] for x in batch] == [x['request_id'] for x in results],
            'Continuation batch must preserve exact IDs/count/order')
    require(len({x['request_id'] for x in imports}) == len(imports), 'Duplicate imported request')
    require(len({x['split'] for x in batch}) == 1, 'Train/dev shared generation context')
    require(dispatch['reasoning_effort'] is None and job['reasoning_effort'] is None, 'Unrequested effort override')
    for request, result, imported, validation in zip(batch, results, imports, receipt['validation']):
        rid = request['request_id']
        require(rid in requests and requests[rid] == request, 'Changed or unknown original request')
        response = synth._normal_response(imported['response'], request)
        execution = response['execution']
        require(response['text'] == result['text'] and set(result) == {'request_id', 'text'}, 'Teacher text changed')
        require(execution['input_request_sha256'] == synth._hash(request)
                and execution['batch_id'] == bid and execution['task_path'] == execution['agent_id'] == dispatch['task_path'],
                'Execution input/context binding differs')
        for key in ('raw_output_sha256', 'wrapper_sha256', 'batch_input_sha256'):
            require(execution[key] == receipt[key], 'Execution hash differs')
        require(synth._validate_response(request, response) == validation, 'Sealed validation differs')
    require(len(receipt['validation']) == len(imports), 'Incomplete receipt validation')
    return imports, bid


def audit(root, package, amendment_paths):
    """Read-only audit, also suitable before or after a derived publication."""
    root = Path(root).resolve()
    require(amendment_paths, 'At least one frozen continuation is required')
    synth = module(package)
    current = state(root, synth)
    run, _, items, _, assigned = current
    amendments = {}
    overrides = {}
    for path in amendment_paths:
        amendment, _ = verify_amendment(root, synth, path, current)
        sha = amendment['amendment_sha256']
        require(sha not in amendments, 'Duplicate amendment')
        require(not set(overrides).intersection(amendment['request_overrides']), 'Overlapping amendments')
        amendments[sha] = amendment
        overrides.update({rid: sha for rid in amendment['request_overrides']})
    continuation, standard = [], []
    for item in items:
        rid, attempt, origin = item['request_id'], item['attempt'], item['origin']
        if origin.startswith(ORIGIN_PREFIX):
            sha = origin[len(ORIGIN_PREFIX):]
            require(sha in amendments and overrides.get(rid) == sha and attempt == 2,
                    'Unamended/nonwhitelisted/over-cap continuation attempt')
            amendment = amendments[sha]
            bid = item['response']['execution']['batch_id']
            require(bid not in amendment['pre_amendment_dispatches'], 'Extra generation was dispatched before amendment freeze')
            imports, _ = bind_import(root, synth, root / 'codex_imports' / f'{bid}.jsonl', current[3])
            require(any(r['request_id'] == rid and r['response'] == item['response'] for r in imports), 'Attempt missing original sealed response')
            continuation.append({'request_id': rid, 'attempt': attempt, 'record_sha256': item['record_sha256'],
                                 'amendment_sha256': sha, 'accepted': item['validation']['accepted']})
        else:
            require(origin == 'external_import' and attempt <= 1, 'Unamended retry or unexpected attempt origin')
            standard.append(item)
    require(len({x['request_id'] for x in continuation}) == len(continuation), 'More than one extra attempt per request')
    return {'schema_version': 1, 'protocol': PROTOCOL, 'original_run_sha256': run['run_sha256'],
            'original_config_sha256': run['config_sha256'], 'adapter_sha256': digest(Path(__file__)),
            'original_max_retries': 1, 'effective_budget': {'max_calls': 1920, 'default_max_retries': 1,
                'request_max_retries': {rid: 2 for rid in sorted(overrides)}},
            'amendment_sha256': sorted(amendments), 'standard_import_attempts': len(standard),
            'continuation_import_attempts': len(continuation), 'total_import_attempts': len(items),
            'total_dispatched_request_assignments': sum(v['assigned_requests'] for v in assigned.values()),
            'accepted_tasks': len(synth._accepted(items)), 'expected_tasks': run['expected_tasks'],
            'continuation_attempts': continuation, 'teacher_text_modified': False, 'validator_modified': False,
            'request_namespace': 'unchanged_original_frozen_run',
            'scope': 'artifact/policy consistency; not independent model attestation or mathematical review'}


def import_responses(root, package, amendment_path, responses_path):
    root = Path(root).resolve()
    synth = module(package)
    with synth._lock(root / 'synthesis'):
        paths = [p for p in (root / 'codex_continuations').glob('*.json') if read(p).get('protocol') == PROTOCOL]
        audit(root, package, paths)  # Reject an invalid existing ledger before appending anything.
        amendment, current = verify_amendment(root, synth, amendment_path)
        run, pool, items, requests, _ = current
        imports, bid = bind_import(root, synth, responses_path, requests)
        require(bid not in amendment['pre_amendment_dispatches'], 'Extra generation must be dispatched after amendment freeze')
        pending = []
        origin = ORIGIN_PREFIX + amendment['amendment_sha256']
        for row in imports:
            rid = row['request_id']
            require(rid in amendment['request_overrides'], 'Imported request is not whitelisted')
            response = synth._normal_response(row['response'], requests[rid])
            sha = synth._hash(response)
            prior = sorted((i for i in items if i['request_id'] == rid), key=lambda i: i['attempt'])
            if len(prior) == 3:
                require(prior[-1]['attempt'] == 2 and prior[-1]['origin'] == origin
                        and prior[-1]['response_sha256'] == sha, 'Continuation retry cap exhausted or response replacement attempted')
                continue  # Exact re-import is idempotent; never rewrite a ledger record.
            exhausted(requests[rid], prior)
            require(not (root / 'synthesis/calls' / rid / '0002.json').exists(), 'New attempt path already exists')
            pending.append({'schema_version': 1, 'request_id': rid, 'attempt': 2, 'origin': origin,
                            'status': 'completed', 'response': response, 'response_sha256': sha,
                            'validation': synth._validate_response(requests[rid], response)})
        require(len(items) + len(pending) <= 1920, 'Global import budget exceeded')
        # Source stays frozen; origin binding enters the record before its first
        # atomic write. There is no transient unamended over-budget ledger entry.
        for record in pending:
            synth._write_attempt(root / 'synthesis', record)
        refreshed, refreshed_items, accepted = synth._refresh(root / 'synthesis', run, pool)
        synth._atomic(root / 'synthesis/synthesis_status.json', synth._status(run, refreshed, refreshed_items, accepted))
    # Reports include every frozen sidecar, so a future disjoint amendment is
    # auditable without obscuring the original standard-import accounting.
    paths = [p for p in (root / 'codex_continuations').glob('*.json') if read(p).get('protocol') == PROTOCOL]
    return audit(root, package, paths)


def publish(root, package, amendment_paths, internal_data_out=None, data_out=None):
    root = Path(root).resolve()
    internal = Path(internal_data_out).resolve() if internal_data_out else root / 'synthesis/base_publication'
    target = Path(data_out).resolve() if data_out else root / 'synthesis/data'
    require(internal != target, 'Original and derived publication paths must differ')
    require(internal not in target.parents and target not in internal.parents, 'Publication directories cannot contain one another')
    synth = module(package)
    report = audit(root, package, amendment_paths)
    require(report['accepted_tasks'] == report['expected_tasks'], 'Continuation synthesis incomplete; no derived publication')
    # The baseline engine runs unchanged and keeps its exact original manifest.
    baseline = synth.finalize(root / 'synthesis', data_out=internal)
    before_ledger = ledger_snapshot(root / 'synthesis')
    original_manifest_sha = digest(internal / 'manifest.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.continuation-publishing-', dir=target.parent))
    try:
        for name in (*baseline['sha256'], 'manifest.json'):
            dest = staging / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(internal / name, dest)
        additions = {}
        for path in amendment_paths:
            amendment = read(path)
            name = f"provenance/{amendment['amendment_id']}.json"
            dest = staging / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            additions[name] = digest(dest)
        provenance = staging / 'provenance'
        provenance.mkdir(exist_ok=True)
        shutil.copyfile(internal / 'manifest.json', provenance / 'base_manifest.json')
        shutil.copyfile(Path(__file__), provenance / 'continuation.py')
        immutable_json(provenance / 'continuation_report.json', report)
        for name in ('base_manifest.json', 'continuation.py', 'continuation_report.json'):
            additions['provenance/' + name] = digest(provenance / name)
        manifest = copy.deepcopy(baseline)
        manifest.pop('manifest_content_sha256')
        manifest['publication_protocol'] = PROTOCOL
        manifest['continuation'] = {**report, 'original_publication_manifest_sha256': original_manifest_sha,
            'original_budget_status': baseline['synthesis']['budget'],
            'original_config_preserved': True,
            'explanation': 'Original request/run namespace and validator are unchanged; named requests received one disclosed extra format retry. Original import_attempt_records excludes continuation origins; use the separate combined totals.'}
        manifest['sha256'].update(additions)
        manifest['manifest_content_sha256'] = synth._hash(manifest)
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        # Use the actual training loader before making the derived directory visible.
        from src.verifiable.expert_train import read_dataset
        for role in ('extractor', 'reasoner', 'verifier'):
            read_dataset(staging, role)
        require(ledger_snapshot(root / 'synthesis') == before_ledger, 'Ledger changed during publication')
        if target.exists() and any(target.iterdir()):
            require(read(target / 'manifest.json') == manifest
                    and all(digest(target / name) == sha for name, sha in manifest['sha256'].items()),
                    'Existing derived publication differs')
        else:
            if target.exists():
                target.rmdir()
            staging.rename(target)
        return {'data_out': str(target), 'internal_data_out': str(internal),
                'manifest_sha256': digest(target / 'manifest.json'), 'continuation': report}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze_p = sub.add_parser('freeze')
    freeze_p.add_argument('--id', required=True)
    freeze_p.add_argument('--request-id', action='append', required=True)
    import_p = sub.add_parser('import')
    import_p.add_argument('--amendment', type=Path, required=True)
    import_p.add_argument('--responses-jsonl', type=Path, required=True)
    for name in ('audit', 'publish'):
        p = sub.add_parser(name)
        p.add_argument('--amendment', type=Path, action='append', required=True)
        if name == 'publish':
            p.add_argument('--internal-data-out', type=Path)
            p.add_argument('--data-out', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze(args.root, args.package, args.id, args.request_id)
    elif args.command == 'import':
        result = import_responses(args.root, args.package, args.amendment, args.responses_jsonl)
    elif args.command == 'audit':
        result = audit(args.root, args.package, args.amendment)
    else:
        result = publish(args.root, args.package, args.amendment, args.internal_data_out, args.data_out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
