"""Create lossless messages-style JSON views; canonical trainer keeps using sft/."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def view(row):
    return {'messages': [*row['prompt'], {'role': 'assistant', 'content': row['response']}],
            'metadata': {k: row[k] for k in ('role', 'split', 'question_hash', 'teacher_request_id',
                                           'teacher_model', 'teacher_generation_surface', 'reviewed')}}


def main():
    (ROOT / 'json').mkdir(exist_ok=True)
    for role in ('extractor', 'reasoner', 'verifier'):
        for split in ('train', 'dev'):
            rows = [json.loads(s) for s in (ROOT / 'sft' / role / f'{split}.jsonl').read_text().splitlines() if s]
            content = json.dumps([view(row) for row in rows], ensure_ascii=False, indent=2) + '\n'
            target = ROOT / 'json' / f'{role}.{split}.json'
            if target.exists() and target.read_text() != content:
                raise ValueError(f'Refusing to overwrite a different export: {target}')
            target.write_text(content)
            print(f'{target.name}: {len(rows)}')


if __name__ == '__main__':
    main()
