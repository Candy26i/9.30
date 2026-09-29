"""Expert-first controller: isolated data -> three independent SFTs -> reload -> freeze.

This entry point never starts Manager training or external benchmark evaluation.
The persistent deadline limits this controller's children, not RunPod billing.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .protocol import KINDS, advisor_messages
from .telemetry import Monitor, atomic_json

PACKAGE = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(config, manager_data_dir, output, raw_jsonl=None):
    root = Path(output).resolve()
    cfg = read(config)
    command = [sys.executable, '-m', 'src.verifiable.expert_data', '--out', str(root / 'data'),
               '--manager-data-dir', str(Path(manager_data_dir).resolve()),
               '--train-size', str(cfg.get('train_size', 128)), '--dev-size', str(cfg.get('dev_size', 32)),
               '--scan-limit', str(cfg.get('scan_limit', 30000)), '--seed', str(cfg['seed'])]
    if raw_jsonl:
        command += ['--raw-jsonl', str(Path(raw_jsonl).resolve())]
    plan = [{'name': 'data', 'command': command}]
    for role in KINDS:
        plan.append({'name': 'sft_' + role, 'command': [sys.executable, '-m', 'src.verifiable.expert_train',
            '--config', str(root / 'config.json'), '--data-dir', str(root / 'data'), '--role', role,
            '--output', str(root / 'training' / role), '--resume']})
    plan.append({'name': 'reload_smoke', 'command': [sys.executable, '-m', 'src.verifiable.experts', 'smoke',
        '--bundle', str(root / 'experts.json'), '--out', str(root / 'reload_smoke')]})
    return plan


def gpu_preflight(gpu):
    """Single-GPU SFT needs no advisor GPU; reject an occupied or missing GPU."""
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA GPU to expert SFT with --gpu')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The 9B pilot requires a BF16-capable CUDA GPU')
    free, total = torch.cuda.mem_get_info(0)
    info = {'physical_gpu': gpu, 'device': torch.cuda.get_device_name(0),
            'free_bytes': free, 'total_bytes': total, 'bf16_supported': True,
            'note': 'A capacity check is not a 9B OOM guarantee. Recommended: one 80 GB GPU.'}
    if free < 40 * 1024**3:
        raise RuntimeError('Expert pilot requires at least 40 GiB free GPU memory; use an idle 80 GB GPU')
    return info


def stop_child(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_child(step, root, deadline, monitor):
    log = root / 'logs' / (step['name'] + '.log')
    if time.time() >= deadline:
        raise TimeoutError('Original two-hour expert budget expired; checkpoints retained')
    command = list(step['command'])
    if step['name'] == 'data' and (root / 'data/manifest.json').exists():
        command.append('--resume')
    with log.open('a') as stream:
        child = subprocess.Popen(command, cwd=PACKAGE, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        try:
            while child.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError('Expert wall-time budget exhausted; checkpoints retained')
                monitor.update(current_stage=step['name'], remaining_seconds=max(0, deadline-time.time()))
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(f"{step['name']} exited {child.returncode}; see {log}")
        finally:
            stop_child(child)


def export_bundle(root, cfg):
    from .runner import checkpoint_identity
    from .serve import load_expert_bundle
    from .expert_train import train_expert
    root = Path(root)
    roles, summaries = {}, {}
    for role in KINDS:
        path = root / 'training' / role
        summary = read(path / 'summary.json')
        if (summary.get('training_complete') is not True or summary.get('role') != role
                or summary.get('optimizer_steps') != cfg['max_steps']
                or summary.get('base_model') != cfg['base_model']
                or summary.get('base_model_revision') != cfg['base_model_revision']
                or summary.get('data_manifest_sha256') != digest(root / 'data' / 'manifest.json')):
            raise ValueError(f'{role} is not a complete, matching expert training result')
        adapter = path / 'adapter_model.safetensors'
        if not adapter.exists():
            adapter = path / 'adapter_model.bin'
        if digest(adapter) != summary.get('adapter_sha256'):
            raise ValueError(f'{role} adapter differs from the completed training result')
        # The completed fast path verifies config/tokenizer exports, the training
        # signature, all six data files and optimizer/scheduler/RNG checkpoints.
        # It cannot launch training here: completion was checked immediately above.
        validated = train_expert(cfg, root / 'data', role, path, resume=True)
        if validated != summary:
            raise ValueError(f'{role} completed training changed during export')
        roles[role] = {'checkpoint': 'training/' + role, 'identity': checkpoint_identity(str(path))}
        summaries[role] = summary
    bundle = {'schema_version': 1, 'frozen': True, 'base_model': cfg['base_model'],
              'base_model_revision': cfg['base_model_revision'],
              'template_sha256': digest(Path(__file__).with_name('chat_template.jinja')), 'roles': roles}
    path = root / 'experts.json'
    if path.exists() and read(path) != bundle:
        raise ValueError('Existing frozen experts changed; use a new experiment directory')
    atomic_json(path, bundle)
    load_expert_bundle(path, base_model=cfg['base_model'], revision=cfg['base_model_revision'])
    return summaries


def manager_config(source, bundle_path):
    from .runner import load_config
    from .serve import load_expert_bundle
    cfg = load_config(source)
    bundle = load_expert_bundle(bundle_path)
    if (cfg['base_model'], cfg.get('base_model_revision')) != (bundle['base_model'], bundle['base_model_revision']):
        raise ValueError('Manager and expert base/revision differ')
    cfg['advisor_expert_bundle'] = str(Path(bundle_path).resolve())
    data_manifest = Path(bundle_path).resolve().parent / 'data/manifest.json'
    cfg['expert_data_manifest'] = str(data_manifest)
    cfg['expert_data_manifest_sha256'] = digest(data_manifest)
    cfg['advisor_models'] = {role: role for role in KINDS}
    return cfg


def smoke(bundle_path, output):
    from ..benchmarks.base import StandardRow
    from .serve import FrozenExpertBackend, load_expert_bundle, generate_advisor_request
    from .provenance import harness_identity
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    bundle = load_expert_bundle(bundle_path)
    signature = {'bundle': bundle, 'harness': harness_identity(), 'purpose': 'adapter_reload_and_role_routing_only'}
    if (root / 'run.json').exists():
        if read(root / 'run.json') != signature:
            raise ValueError('Reload smoke inputs changed')
        if (root / 'summary.json').exists() and read(root / 'summary.json').get('reload_complete'):
            return read(root / 'summary.json')
    elif any(root.iterdir()):
        raise ValueError('Nonempty reload directory without manifest')
    atomic_json(root / 'run.json', signature)
    with Monitor(root, 'expert_reload') as monitor:
        backend = FrozenExpertBackend(bundle, max_context=8192)
        row = StandardRow(0, 'synthetic_reload', 'free_response_math',
                          'A box has 2 red balls and 3 blue balls. How many balls are in the box?', {}, '5')
        records = []
        # Repeat the first adapter after both others to expose switching bugs.
        for role in (*KINDS, KINDS[0]):
            messages = advisor_messages(role, row, '2 + 3 = 5.')
            result, settings = generate_advisor_request(backend,
                {'model': role, 'messages': messages, 'max_tokens': 128, 'temperature': 0.0, 'seed': 42})
            actual = result.get('margent_role', {})
            if actual.get('role') != role or actual.get('identity') != bundle['roles'][role]['identity']:
                raise RuntimeError('Expert role routing identity failed on GPU')
            if result.get('error') or not result.get('text', '').strip():
                raise RuntimeError(f'{role} reload produced no usable text')
            monitor.generation(role, result, messages=messages)
            records.append({'role': role, 'actual': actual, 'truncated': result.get('truncated', False)})
        result = {'reload_complete': True, 'requests': records,
                  'scope': 'Weights reload and role routing only; not evidence of expert quality or benchmark improvement.'}
        atomic_json(root / 'summary.json', result)
        monitor.summary(result)
        return result


def run(args):
    from .data import verify_manifest
    from .provenance import harness_identity
    root = Path(args.out).resolve()
    cfg = read(args.config)
    if not 0 < args.minutes <= 120:
        raise ValueError('Expert pilot budget must be in (0, 120] minutes')
    plan = build_plan(args.config, args.manager_data_dir, root, args.raw_jsonl)
    signature = {'schema_version': 1, 'purpose': 'expert_sft_before_manager', 'config': cfg,
        'manager_manifest': verify_manifest(args.manager_data_dir),
        'manager_config': read(args.manager_config), 'harness': harness_identity(),
        'raw_sha256': digest(args.raw_jsonl) if args.raw_jsonl else None,
        'minutes': args.minutes, 'gpu': args.gpu, 'plan': plan}
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        meta = root / 'expert_run.json'
        if meta.exists():
            if read(meta) != signature or read(root / 'config.json') != cfg:
                raise ValueError('Expert code, data or settings changed; use a new output directory')
        else:
            if any(p.name != '.controller.lock' for p in root.iterdir()):
                raise ValueError('Nonempty expert directory without matching manifest')
            atomic_json(root / 'config.json', cfg)
            atomic_json(root / 'budget.json', {'deadline_unix': time.time()+args.minutes*60})
            atomic_json(meta, signature)
        deadline = read(root / 'budget.json')['deadline_unix']
        (root / 'logs').mkdir(exist_ok=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        stage = 'preflight'
        with Monitor(root, 'expert_sft_controller') as monitor:
            try:
                monitor.summary({'controller_status': 'running', 'experts_complete': False,
                                 'current_stage': stage, 'failed_stage': None, 'deadline_unix': deadline})
                if (root / 'expert_report.json').exists():
                    summaries = export_bundle(root, cfg)
                    report = read(root / 'expert_report.json')
                    expected_manager = manager_config(args.manager_config, root / 'experts.json')
                    if read(root / 'manager_config.json') != expected_manager:
                        raise ValueError('Frozen Manager config changed after expert completion')
                    if report.get('experts_complete') is not True or report.get('controller_status') != 'completed':
                        raise ValueError('Existing expert report does not certify completed training')
                    from .serve import load_expert_bundle
                    smoke_signature = {'bundle': load_expert_bundle(root / 'experts.json'),
                        'harness': harness_identity(), 'purpose': 'adapter_reload_and_role_routing_only'}
                    if read(root / 'reload_smoke' / 'run.json') != smoke_signature:
                        raise ValueError('Reload smoke provenance differs from frozen experts')
                    if report.get('roles') != summaries:
                        raise ValueError('Expert report differs from completed role results')
                    if not read(root / 'reload_smoke' / 'summary.json').get('reload_complete'):
                        raise ValueError('Expert report lacks a completed reload smoke')
                    atomic_json(root / 'expert_status.json', report)
                    monitor.summary(report)
                    return report
                if time.time() >= deadline:
                    raise TimeoutError('Original expert budget expired; no automatic restart or budget extension')
                atomic_json(root / 'gpu_preflight.json', gpu_preflight(args.gpu))
                for step in plan:
                    stage = step['name']
                    monitor.summary({'current_stage': stage})
                    if stage == 'reload_smoke':
                        export_bundle(root, cfg)
                    run_child(step, root, deadline, monitor)
                    if stage == 'data':
                        # Metadata only. Text lives in the separate opt-in data artifact.
                        atomic_json(root / 'expert_data_report.json', read(root / 'data' / 'manifest.json'))
                summaries = export_bundle(root, cfg)
                manager = manager_config(args.manager_config, root / 'experts.json')
                atomic_json(root / 'manager_config.json', manager)
                report = {'experts_complete': True, 'controller_status': 'completed', 'current_stage': 'complete',
                    'failed_stage': None, 'roles': summaries, 'expert_bundle': str(root / 'experts.json'),
                    'manager_config': str(root / 'manager_config.json'), 'manager_started': False,
                    'quality_validation': 'weak-supervision pilot; dev loss and reload checked; manual role review and downstream helpfulness still required',
                    'pod_billing_stopped': False}
                atomic_json(root / 'expert_report.json', report)
                atomic_json(root / 'expert_status.json', report)
                monitor.summary(report)
                return report
            except BaseException as exc:
                state = 'budget_exhausted' if isinstance(exc, TimeoutError) else 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
                report = {'experts_complete': False, 'controller_status': state, 'current_stage': stage,
                          'failed_stage': stage, 'error': str(exc), 'pod_billing_stopped': False}
                atomic_json(root / 'expert_status.json', report)
                (root / 'controller_traceback.txt').write_text(traceback.format_exc())
                monitor.summary(report)
                raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    for op in ('plan', 'run'):
        s = sub.add_parser(op)
        s.add_argument('--config', default=str(PACKAGE / 'configs/math_expert_sft_pilot.json'))
        s.add_argument('--manager-config', default=str(PACKAGE / 'configs/math_rsi_actions.json'))
        s.add_argument('--manager-data-dir', default='/workspace/margent-data-restart-20260925')
        s.add_argument('--out', default='/workspace/margent-expert-sft-01')
        s.add_argument('--raw-jsonl')
        s.add_argument('--minutes', type=float, default=120)
        s.add_argument('--gpu', default='0')
    s = sub.add_parser('smoke')
    s.add_argument('--bundle', required=True)
    s.add_argument('--out', required=True)
    s = sub.add_parser('serve')
    s.add_argument('--config', required=True)
    args = p.parse_args()
    if args.operation == 'plan':
        result = build_plan(args.config, args.manager_data_dir, args.out, args.raw_jsonl)
    elif args.operation == 'smoke':
        result = smoke(args.bundle, args.out)
    elif args.operation == 'serve':
        from urllib.parse import urlsplit
        from .runner import load_config
        cfg = load_config(args.config)
        address = urlsplit(cfg['advisor_url'])
        if address.hostname not in ('127.0.0.1', 'localhost') or address.scheme != 'http':
            raise ValueError('The owned expert service must use loopback HTTP')
        command = [sys.executable, '-m', 'src.verifiable.serve', '--model', cfg['base_model'],
            '--revision', cfg['base_model_revision'], '--max-context', str(cfg['max_context']),
            '--port', str(address.port or 80)]
        if cfg.get('advisor_expert_bundle'):
            command += ['--expert-bundle', cfg['advisor_expert_bundle']]
        os.execv(sys.executable, command)
    else:
        result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
