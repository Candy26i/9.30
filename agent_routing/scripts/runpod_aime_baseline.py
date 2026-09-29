"""Bounded, resumable AIME baseline: three dev questions, then all 30 test questions."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from runpod_rsi_smoke import gpu_check, stop_owned
from src.verifiable.telemetry import Monitor, atomic_json, metrics, progress


def read(path):
    return json.loads(Path(path).read_text())


def shards(stage):
    return {p.name: p.read_bytes() for p in (stage / 'questions').glob('*.json')}


def verify_complete(stage, expected):
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Expected question identities must be nonempty and unique')
    records = [json.loads(line) for line in (stage / 'records.jsonl').read_text().splitlines() if line.strip()]
    if len(records) != len(expected) or {r['question_hash'] for r in records} != set(expected):
        raise ValueError('Evaluation did not finish every expected question exactly once')
    if read(stage / 'summary.json')['n'] != len(expected):
        raise ValueError('Summary count does not match the complete benchmark')
    # When question checkpoints exist they are authoritative for resume. Do not
    # certify a stale aggregate file that a later restart would overwrite.
    if (stage / 'questions').exists():
        saved = {p.stem: read(p) for p in (stage / 'questions').glob('*.json')}
        if set(saved) != set(expected) or any(saved.get(r['question_hash']) != r for r in records):
            raise ValueError('Question checkpoints disagree with completed aggregate records')
    return records


def run_child(command, stage, log, env, deadline, advisor=None, interrupt_after_one=False):
    """Only terminate our own child group. Persisted question shards survive interruption."""
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        last = -1
        try:
            while process.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError('Two-hour run budget exhausted; saved questions are retained')
                if advisor is not None and advisor.poll() is not None:
                    raise RuntimeError('Owned advisor exited; see logs/advisor.log')
                count = len(shards(stage))
                if count != last:
                    progress(completed_questions=count)
                    print(f'[{stage.name}] {count} questions saved', flush=True)
                    last = count
                if interrupt_after_one and count:
                    stop_owned(process)
                    return True
                time.sleep(.5)
            if process.returncode:
                raise RuntimeError(f'{stage.name} exited {process.returncode}; see {log}')
            return False
        except BaseException:
            stop_owned(process)
            raise


def preflight_stats(records, strict=False):
    invalid = sum(not r['direct_valid'] or not r['policy']['valid'] for r in records)
    truncated = sum(any(c.get('truncated', False) for c in r.get('costs', []) if c.get('role') == 'manager') for r in records)
    warnings = []
    if invalid >= 2:
        warnings.append('Most dev questions had invalid output; these count as incorrect in evaluation')
    if truncated >= 2:
        warnings.append('Most dev questions reached a Manager token budget')
    return {'n': len(records), 'invalid_questions': invalid, 'truncated_questions': truncated,
            'strict': strict, 'warnings': warnings, 'proceed': not strict or not warnings,
            'note': 'Output quality is descriptive by default, not an infrastructure failure. Strict quality gate is opt-in. Accuracy is never a gate; AIME settings remain frozen.'}


def initialize(args):
    from src.verifiable.data import identity, load_rows, verify_manifest
    from src.verifiable.provenance import harness_identity
    from src.verifiable.runner import load_config
    root = Path(args.out).resolve()
    source = Path(args.data_dir).resolve()
    manifest = verify_manifest(str(source))
    cfg = load_config(args.config)
    cfg['advisor_url'] = f'http://127.0.0.1:{args.port}'
    dev = sorted(load_rows(str(source / 'dev.jsonl'), required_split='dev'), key=lambda r: identity(r.question))[:3]
    test = load_rows(str(source / 'aime2026.jsonl'), required_split='test')
    if len(dev) != 3 or len(test) != 30:
        raise ValueError('Expected three dev and the full 30-question AIME2026 set')
    dev_ids, test_ids = [identity(r.question) for r in dev], [identity(r.question) for r in test]
    if len(set(test_ids)) != 30 or set(dev_ids) & set(test_ids):
        raise ValueError('Duplicate AIME questions or dev/test overlap')
    content = ''.join(json.dumps(r.to_dict(), ensure_ascii=False) + '\n' for r in dev)
    signature = {'purpose': 'locked_initial_aime_baseline_no_training', 'config': cfg,
                 'source_manifest': manifest, 'harness': harness_identity(), 'data_dir': str(source),
                 'dev_ids': dev_ids, 'test_ids': test_ids, 'minutes': args.minutes,
                 'strict_preflight': getattr(args, 'strict_preflight', False),
                 'manager_gpu': args.manager_gpu, 'advisor_gpu': args.advisor_gpu}
    file = root / 'benchmark_run.json'
    if file.exists():
        if read(file) != signature:
            raise ValueError('Code, data or settings changed; keep the recorded version or use a new run directory')
        if (root / 'dev.jsonl').read_text() != content or read(root / 'config.json') != cfg:
            raise ValueError('Frozen local evaluation inputs changed')
    else:
        if any(p.name != '.controller.lock' for p in root.iterdir()):
            raise ValueError('Nonempty baseline directory without matching manifest')
        atomic_json(file, signature)
        atomic_json(root / 'config.json', cfg)
        (root / 'dev.jsonl').write_text(content)
        atomic_json(root / 'budget.json', {'deadline_unix': time.time() + args.minutes * 60})
    (root / 'logs').mkdir(exist_ok=True)
    return root, source, cfg, dev_ids, test_ids, read(root / 'budget.json')['deadline_unix']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='/workspace/margent-aime-baseline-01')
    p.add_argument('--data-dir', default='/workspace/margent-data-restart-20260925')
    p.add_argument('--config', default=str(REPO / 'configs/math_rsi_actions.json'))
    p.add_argument('--minutes', type=float, default=120)
    p.add_argument('--manager-gpu', default='1')
    p.add_argument('--advisor-gpu', default='0')
    p.add_argument('--port', type=int, default=8003)
    p.add_argument('--strict-preflight', action='store_true',
                   help='Stop when at least two dev outputs are invalid/truncated; default records warnings and evaluates all questions')
    args = p.parse_args()
    if not 0 < args.minutes <= 120:
        p.error('--minutes must be in (0, 120]')
    os.environ.update(HF_HOME='/workspace/hf-cache', HF_HUB_CACHE='/workspace/hf-cache/hub',
        HF_DATASETS_CACHE='/workspace/hf-cache/datasets', HF_HUB_DISABLE_XET='1',
        TMPDIR='/workspace/margent-tmp', MARGENT_WANDB_MODE='online', MARGENT_WANDB_TEXT='1')
    os.environ.setdefault('WANDB_ENTITY', 'yuningyangaillm')
    os.environ.setdefault('WANDB_PROJECT', 'MATH_rsi')
    Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.controller.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This baseline already has a running controller')
    root, source, cfg, dev_ids, test_ids, deadline = initialize(args)
    if (root / 'baseline_report.json').exists():
        completed = read(root / 'baseline_report.json')
        verify_complete(root / 'aime2026', test_ids)
        if completed.get('status') != 'completed' or completed.get('n') != len(test_ids):
            raise ValueError('Existing baseline report is not a complete benchmark result')
        print(json.dumps(completed, indent=2)); return
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.manager_gpu, 'PYTHONUNBUFFERED': '1'}
    advisor = None
    current = 'preflight'
    with Monitor(root, 'aime_baseline') as monitor:
        try:
            monitor.summary({'baseline_complete': False, 'controller_status': 'running',
                             'current_stage': current, 'failed_stage': None, 'error': None,
                             'planned_dev_interruption': False, 'expected_questions': len(test_ids),
                             'deadline_unix': deadline})
            if time.time() >= deadline:
                raise TimeoutError('Original budget expired; no automatic budget extension or GPU restart')
            atomic_json(root / 'gpu_preflight.json', gpu_check(args.manager_gpu, args.advisor_gpu))
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', args.port))
            current = 'advisor_start'
            monitor.summary({'current_stage': current, 'controller_status': 'running'})
            progress(phase=current, total_questions=30)
            with (root / 'logs/advisor.log').open('a') as stream:
                advisor = subprocess.Popen([sys.executable, '-m', 'src.verifiable.experts', 'serve',
                    '--config', str(root / 'config.json')],
                    cwd=REPO, env={**env, 'CUDA_VISIBLE_DEVICES': args.advisor_gpu}, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
            ready_limit = min(deadline, time.time() + 300)
            while True:
                if advisor.poll() is not None or time.time() >= ready_limit:
                    raise RuntimeError('Advisor startup failed or exceeded five minutes; see logs/advisor.log')
                try:
                    with urlopen(cfg['advisor_url'] + '/health', timeout=2) as response:
                        if json.load(response).get('status') == 'ready':
                            break
                except OSError:
                    pass
                time.sleep(1)

            def command(stage, data, kind):
                entry = ['-m', 'src.verifiable.rsi', 'stage', 'assess'] if kind == 'dev' else ['-m', 'src.verifiable', 'evaluate', '--resume']
                return [sys.executable, *entry, '--config', str(root / 'config.json'), '--checkpoint', cfg['base_model'],
                        '--data', str(data), '--out', str(stage)]

            current = 'dev_preflight_and_resume_check'
            monitor.summary({'current_stage': current, 'planned_dev_interruption': True})
            progress(phase=current, total_questions=3, completed_questions=0)
            dev = root / 'dev_preflight'
            cmd = command(dev, root / 'dev.jsonl', 'dev')
            drill = root / 'resume_check.json'
            # Planned interruption only once, before reading any AIME outputs.
            if not drill.exists():
                interrupted = run_child(cmd, dev, root / 'logs/dev_preflight.log', env, deadline, advisor, True)
                preserved = shards(dev)
                atomic_json(drill, {'interrupted': interrupted, 'saved_sha256': {
                    k: hashlib.sha256(v).hexdigest() for k, v in preserved.items()}, 'verified': False})
            run_child(cmd, dev, root / 'logs/dev_preflight.log', env, deadline, advisor)
            saved = read(drill)
            for name, digest in saved['saved_sha256'].items():
                if hashlib.sha256((dev / 'questions' / name).read_bytes()).hexdigest() != digest:
                    raise ValueError('Resume changed a previously completed question')
            records = verify_complete(dev, dev_ids)
            saved['verified'] = True
            atomic_json(drill, saved)
            monitor.summary({'planned_dev_interruption': False, 'resume_check': saved})
            check = preflight_stats(records, strict=args.strict_preflight)
            # A rough dev-based estimate is not a promise for harder AIME items.
            seconds = [sum(c.get('seconds', 0) for c in r.get('costs', [])) for r in records]
            check.update(estimated_aime_minutes_from_dev=sum(seconds) / 3 * 30 / 60,
                         remaining_minutes=max(0, deadline-time.time()) / 60,
                         estimate_note='Dev-only generation timing; AIME can be slower. Hard deadline takes precedence.')
            atomic_json(root / 'preflight_report.json', check)
            metrics(check, 'preflight')
            print(json.dumps(check, indent=2), flush=True)
            if not check['proceed']:
                raise RuntimeError('At least two of three dev questions had invalid or truncated output; inspect preflight before spending on AIME')
            if check['estimated_aime_minutes_from_dev'] * 1.25 > check['remaining_minutes'] and not shards(root / 'aime2026'):
                raise RuntimeError('Dev timing estimate plus 25% margin exceeds remaining budget; AIME has not started. Review preflight_report.json')
            current = 'aime2026_all_30'
            monitor.summary({'current_stage': current})
            progress(phase=current, total_questions=30, completed_questions=len(shards(root / 'aime2026')))
            stage = root / 'aime2026'
            run_child(command(stage, source / 'aime2026.jsonl', 'test'), stage, root / 'logs/aime2026.log', env, deadline, advisor)
            records = verify_complete(stage, test_ids)
            report = {**read(stage / 'summary.json'), 'status': 'completed', 'scope': 'full 30-question baseline under frozen token/decoding budgets; no training or improvement claim',
                'independent_correct_n': sum(r['direct_correct'] for r in records),
                'policy_correct_n': sum(r['policy']['correct'] for r in records),
                'resume_check': saved, 'config': cfg}
            atomic_json(root / 'baseline_report.json', report)
            metrics(report, 'benchmark')
            atomic_json(root / 'baseline_status.json', {'status': 'completed', 'stage': 'complete',
                        'completed_aime_questions': len(records), 'wandb_upload_error': monitor.wandb_failed})
            monitor.summary({'baseline_complete': True, 'baseline_report': report, 'n': len(records),
                             'current_stage': 'complete', 'controller_status': 'completed',
                             'failed_stage': None, 'error': None})
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        except BaseException as exc:
            state = 'budget_exhausted' if isinstance(exc, TimeoutError) else 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            atomic_json(root / 'baseline_status.json', {'status': state, 'stage': current, 'error': str(exc),
                'completed_aime_questions': len(shards(root / 'aime2026')), 'pod_billing_stopped': False})
            (root / 'controller_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
            monitor.summary({'baseline_complete': False, 'controller_status': state, 'failed_stage': current,
                             'error': str(exc), 'planned_dev_interruption': False,
                             'completed_aime_questions': len(shards(root / 'aime2026'))})
            try:
                if monitor.tracker.run is not None:
                    monitor.tracker.run.alert(title='MARGENT AIME baseline stopped', text=f'{state}: {current}: {exc}. Saved outputs: {root}')
            except Exception as alert_error:
                atomic_json(root / 'alert_error.json', {'error': str(alert_error)})
            raise
        finally:
            stop_owned(advisor)
    atomic_json(root / 'baseline_status.json', {**read(root / 'baseline_status.json'),
                'wandb_upload_error': monitor.wandb_failed})


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
