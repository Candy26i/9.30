"""Offline validation of canonical data, source isolation and plain JSON views."""
from pathlib import Path
import hashlib
import json
import sys
from export_json import view

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))


def main():
    from src.verifiable.experts import check_teacher_data
    from src.verifiable.expert_train import read_dataset
    cfg = json.loads((ROOT / 'configs/expert_sft_text_clean.json').read_text())
    manifest, isolation = check_teacher_data(ROOT / 'sft', ROOT / 'manager', cfg)
    total = 0
    for role in ('extractor', 'reasoner', 'verifier'):
        _, train, dev = read_dataset(ROOT / 'sft', role)
        for split, rows in [('train', train), ('dev', dev)]:
            exported = json.loads((ROOT / 'json' / f'{role}.{split}.json').read_text())
            if exported != [view(row) for row in rows]:
                raise ValueError(f'JSON view differs from original role data: {role}/{split}')
            total += len(rows)
    if total != 544:
        raise ValueError('Unexpected retained row count')
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        path.relative_to(ROOT)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Published file changed: {name}')
    print(json.dumps({'valid': True, 'sft_rows': total, 'question_counts': manifest['question_counts'],
                      'exact_json_views': True, 'isolation_checked': isolation['checked'],
                      'gpu_used': False, 'mathematical_quality_reviewed': False}))


if __name__ == '__main__':
    main()
