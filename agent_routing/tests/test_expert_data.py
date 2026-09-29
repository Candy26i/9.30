"""Expert corpora stay disjoint, reproducible, grounded and auditable offline."""
import hashlib
import json

import pytest

from src.benchmarks.base import StandardRow
from src.verifiable import expert_data as ed
from src.verifiable.data import identity
from src.verifiable.protocol import KINDS, advisor_messages


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def manager_data(tmp_path, questions=None):
    root = tmp_path / 'manager'
    files = sorted(ed.MANAGER_FILES)
    questions = questions or [f'Independent locked manager {word} question: evaluate its unique quantity {i}.'
                              for i, word in enumerate(('train', 'dev', 'aime', 'beyond'))]
    hashes, counts = {}, {}
    for i, (name, question) in enumerate(zip(files, questions)):
        split = name.removesuffix('.jsonl') if name in {'train.jsonl', 'dev.jsonl'} else 'test'
        row = StandardRow(i, 'numina', 'free_response_math', question, {}, '7',
                          metadata={'answer_type': 'math', 'content_hash': identity(question)}, split=split)
        path = root / name
        write_jsonl(path, [row.to_dict()])
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[name.removesuffix('.jsonl')] = 1
    (root / 'manifest.json').write_text(json.dumps({'sha256': hashes, 'counts': counts}))
    return root


def record(i):
    words = [f'entity{i}_{j}' for j in range(12)]
    question = (f"Let {words[0]} have {i + 3} units alongside {' '.join(words[1:])}. "
                'Its variables satisfy x = 2 and y = 3. Find the sum x+y.')
    solution = ('The stated values determine the requested sum directly. '
                'For the numeric step, $2 + 3 = 5$. Therefore the sum has value 5. '
                f'Reference annotation for item {i}.')
    return {'problem': question, 'answer': '5', 'solution': solution, 'problem_idx': f'n{i}',
            'problem_is_valid': 'Yes', 'solution_is_valid': 'Yes', 'problem_type': 'algebra',
            'question_type': 'answer'}


@pytest.fixture
def raw_source(tmp_path):
    path = tmp_path / 'raw.jsonl'
    write_jsonl(path, [record(i) for i in range(120)])
    return path


def build(tmp_path, raw_source, **kwargs):
    manager = manager_data(tmp_path)
    out = tmp_path / 'experts'
    manifest = ed.build_expert_data(out, manager, raw_jsonl=raw_source,
                                   train_size=5, dev_size=3, **kwargs)
    return out, manager, manifest


def test_build_all_roles_reproducibly_without_answer_key_in_prompts(tmp_path, raw_source):
    out, manager, manifest = build(tmp_path, raw_source)
    second = tmp_path / 'copy'
    again = ed.build_expert_data(second, manager, raw_jsonl=raw_source, train_size=5, dev_size=3)
    assert manifest == again
    for path in out.rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(out)).read_bytes()
    refs = {r['question_hash']: r for r in read_jsonl(out / 'references.jsonl')}
    assert len(refs) == 8
    assert manifest['question_counts'] == {'train': 5, 'dev': 3}
    assert manifest['counts']['verifier'] == {'train': 10, 'dev': 6}
    assert not manifest['source']['upstream_revision_verified']
    assert not manifest['quality_audit']['reviewed']
    assert manifest['quality_audit']['verifier_verdict_counts']['uncertain'] == 0
    for role in KINDS:
        for split in ('train', 'dev'):
            for row in read_jsonl(out / role / f'{split}.jsonl'):
                ref = refs[row['question_hash']]
                reconstructed = StandardRow(0, 'numina', 'free_response_math', row['question'], {}, 'SECRET_GOLD')
                assert row['prompt'] == advisor_messages(role, reconstructed, row['candidate'])
                assert 'ground_truth' not in row and 'solution' not in row
                assert ref['solution'] not in row['prompt'][1]['content']
                assert row['template_sha256'] == manifest['template_sha256']
                assert row['split'] == ref['split'] == split
                assert not row['reviewed'] and row['teacher_revision'] is None
                if role == 'extractor':
                    assert row['response'] != row['question']
                    assert 'Requested target:' in row['response'] and 'Explicit numeric quantities' in row['response']
                    assert not row['quality_checks']['semantic_equivalent_formulations_generated']
                if role == 'reasoner':
                    assert row['response'] == ref['solution'].strip()
                    assert not row['quality_checks']['derivation_verified']


def test_groups_and_expansion_preserve_membership(tmp_path, raw_source):
    out, manager, _ = build(tmp_path, raw_source)
    bigger = tmp_path / 'expanded'
    ed.build_expert_data(bigger, manager, raw_jsonl=raw_source, train_size=10, dev_size=6)
    seen = {}
    for role in KINDS:
        for split in ('train', 'dev'):
            small = read_jsonl(out / role / f'{split}.jsonl')
            large = read_jsonl(bigger / role / f'{split}.jsonl')
            assert small == large[:len(small)]
            for row in large:
                assert seen.setdefault(row['question_hash'], split) == split
    assert len(seen) == 16


def test_excludes_all_manager_splits_and_near_copies(tmp_path, raw_source):
    manager_questions = [record(i)['problem'] for i in range(4)]
    manager = manager_data(tmp_path, manager_questions)
    records = [record(i) for i in range(120)]
    near = {**record(0), 'problem': record(0)['problem'].replace('Find the sum', 'Please find the sum'), 'problem_idx': 'near'}
    duplicate = {**record(8), 'problem': '  ' + record(8)['problem'].upper() + ' ', 'problem_idx': 'duplicate'}
    records += [near, duplicate]
    write_jsonl(raw_source, records)
    out = tmp_path / 'experts'
    manifest = ed.build_expert_data(out, manager, raw_jsonl=raw_source, train_size=5, dev_size=3)
    refs = read_jsonl(out / 'references.jsonl')
    assert not {identity(q) for q in manager_questions} & {r['question_hash'] for r in refs}
    assert manifest['stats']['manager_exact_overlap'] == 4
    assert manifest['stats']['manager_near_overlap'] == 1
    assert manifest['stats']['source_exact_duplicate'] == 1
    assert {r['reason'] for r in read_jsonl(out / 'exclusions.jsonl')} >= {'manager_exact_overlap', 'manager_near_overlap'}


def test_near_duplicates_in_source_cannot_cross_groups(tmp_path, raw_source):
    records = [record(i) for i in range(120)]
    records.append({**record(10), 'problem': record(10)['problem'].replace('Find the sum', 'Please find the sum'), 'problem_idx': 'near10'})
    write_jsonl(raw_source, records)
    out, _, manifest = build(tmp_path, raw_source)
    assert manifest['stats']['source_near_duplicate'] == 1
    hashes = {r['question_hash'] for r in read_jsonl(out / 'references.jsonl')}
    assert not {identity(records[10]['problem']), identity(records[-1]['problem'])} <= hashes


def test_verifier_corruption_has_exact_executable_evidence(tmp_path, raw_source):
    out, _, _ = build(tmp_path, raw_source)
    for split in ('train', 'dev'):
        grouped = {}
        for row in read_jsonl(out / 'verifier' / f'{split}.jsonl'):
            grouped.setdefault(row['question_hash'], []).append(row)
            left, right = row['candidate'].split('=')
            equal = ed.evaluate_arithmetic(left) == ed.evaluate_arithmetic(right)
            assert equal is (row['verdict'] == 'correct')
            assert row['quality_checks']['arithmetic_verified']
            assert not row['quality_checks']['full_solution_verified']
        for rows in grouped.values():
            assert {r['verdict'] for r in rows} == {'correct', 'incorrect'}
            correct, wrong = sorted(rows, key=lambda r: r['verdict'])
            assert correct['arithmetic_evidence'] == wrong['arithmetic_evidence']
            assert wrong['corruption']['operation'] == 'replace_rhs_with_exact_rhs_plus_one'
            assert 'local check' in correct['response']


@pytest.mark.parametrize('expression', ['__import__("os").system("true")', 'x+1', '1/0', '9**9999', '1<2', '(1,2)', '9'*100])
def test_arithmetic_rejects_unbounded_computation(expression):
    with pytest.raises(ValueError):
        ed.evaluate_arithmetic(expression)


def test_exact_decimals_and_false_reference_steps():
    assert ed.evaluate_arithmetic('0.1 + 0.2') == ed.evaluate_arithmetic('0.3')
    assert ed.evaluate_arithmetic('(3/4)^2') == ed.evaluate_arithmetic('9/16')
    assert ed._arithmetic_step('We obtain $2 + 3 = 6$, which is false.') is None
    assert ed._arithmetic_step('The symbolic equation $x+2+3=5$ is not a pure numeric step.') is None


def test_low_quality_source_does_not_publish_partial_dataset(tmp_path, raw_source):
    manager = manager_data(tmp_path)
    records = [record(i) for i in range(20)]
    for rec in records:
        rec['solution'] = 'The answer is 5.'
    write_jsonl(raw_source, records)
    out = tmp_path / 'experts'
    with pytest.raises(ValueError, match='Not enough eligible'):
        ed.build_expert_data(out, manager, raw_jsonl=raw_source, train_size=2, dev_size=1)
    assert not out.exists()


def test_full_manifest_and_untampered_manager_required(tmp_path, raw_source):
    manager = manager_data(tmp_path)
    path = manager / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['smoke_only'] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='complete frozen Manager'):
        ed.build_expert_data(tmp_path / 'experts', manager, raw_jsonl=raw_source, train_size=2, dev_size=1)
    manifest.pop('smoke_only')
    path.write_text(json.dumps(manifest))
    (manager / 'aime2026.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='Data changed'):
        ed.build_expert_data(tmp_path / 'experts', manager, raw_jsonl=raw_source, train_size=2, dev_size=1)


def test_resume_checks_source_config_outputs_and_manifest_metadata(tmp_path, raw_source):
    out, manager, manifest = build(tmp_path, raw_source)
    kwargs = dict(raw_jsonl=raw_source, train_size=5, dev_size=3, resume=True)
    assert ed.build_expert_data(out, manager, **kwargs) == manifest
    with pytest.raises(ValueError, match='resume request differs'):
        ed.build_expert_data(out, manager, **{**kwargs, 'seed': 123})
    path = out / 'reasoner/train.jsonl'
    contents = path.read_bytes()
    path.write_bytes(contents + b'\n')
    with pytest.raises(ValueError, match='data changed'):
        ed.build_expert_data(out, manager, **kwargs)
    path.write_bytes(contents)
    manifest_path = out / 'manifest.json'
    changed = {**manifest, 'question_counts': {'train': 999, 'dev': 3}}
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='manifest content changed'):
        ed.build_expert_data(out, manager, **kwargs)
    manifest_path.write_text(json.dumps(manifest))
    with raw_source.open('a') as stream:
        stream.write(json.dumps(record(999)) + '\n')
    with pytest.raises(ValueError, match='resume request differs'):
        ed.build_expert_data(out, manager, **kwargs)


def test_pinned_remote_source_and_cli(tmp_path, monkeypatch, capsys):
    manager = manager_data(tmp_path)
    calls = []
    def remote(revision):
        calls.append(revision)
        yield from [record(i) for i in range(120)]
    monkeypatch.setattr(ed, '_remote_rows', remote)
    out = tmp_path / 'experts'
    ed.main(['--out', str(out), '--manager-data-dir', str(manager), '--train-size', '5', '--dev-size', '3'])
    assert calls == [ed.NUMINA_REVISION]
    assert json.loads(capsys.readouterr().out)['question_counts'] == {'train': 5, 'dev': 3}
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['source']['mode'] == 'huggingface_pinned_parquet'
    with pytest.raises(ValueError, match='pinned 40-character'):
        ed.build_expert_data(tmp_path / 'unpinned', manager, source_revision='main')


@pytest.mark.parametrize('missing_image', ['![figure](plot.png)', '<img src="plot.png">',
                                           'As shown in the diagram, we conclude this.',
                                           r'\includegraphics{plot}', r'\begin{tikzpicture}'])
def test_reference_image_dependencies_are_filtered(missing_image):
    raw = record(0)
    row = ed.normalize(raw, 'numina', 0)
    targets, reason = ed._targets(row, raw['solution'] + missing_image, 12000)
    assert targets is None and reason == 'reference_requires_image'


def test_source_ids_and_fallback_provenance_are_explicit(tmp_path, raw_source):
    raw = [record(i) for i in range(120)]
    for i, item in enumerate(raw):
        item.pop('problem_idx')
        if i % 2:
            item['id'] = f'real-source-{i}'
    write_jsonl(raw_source, raw)
    out, _, manifest = build(tmp_path, raw_source)
    for row in read_jsonl(out / 'references.jsonl'):
        i = row['source_ordinal']
        assert row['source_id_kind'] == ('id' if i % 2 else 'pinned_source_stream_index')
        assert row['source_id'] == (f'real-source-{i}' if i % 2 else f'source-index:{i}')
    assert 'fallback IDs require identical source order' in manifest['quality_audit']['source_id_policy']


def test_request_first_extractor_preserves_separate_conditions():
    result = ed._extract_facts('Find the smallest integer n such that n > 20 and n is divisible by 7.')
    assert result is not None
    assert '- such that n > 20 and n is divisible by 7' in result['response']
    assert '- Find the smallest integer n' in result['response']
    assert 'Explicit numeric quantities in the givens: 20, 7' in result['response']


def test_near_index_matches_brute_force_jaccard():
    questions = [record(i)['problem'] for i in range(25)]
    index = ed._NearIndex()
    for i, question in enumerate(questions):
        index.add(question, str(i))
    for question in questions:
        for variant in (question + ' Please.', question.replace('Find', 'Please find'),
                        question.replace('2 and y = 3', '2 and y = 4'), 'entirely unrelated words'):
            features = ed._features(variant)
            expected = any(identity(q) == identity(variant) or (len(features) >= 6 and
                len(ed._features(q)) >= 6 and len(features & ed._features(q)) / len(features | ed._features(q)) >= ed.NEAR_THRESHOLD)
                for q in questions)
            assert bool(index.match(variant)) == expected


@pytest.mark.parametrize('solution', ['$2+3=5 + 7$', '$2+3=5!', '$2+3=5 \\times 2$', '$2+3=5 x$'])
def test_arithmetic_does_not_drop_rhs_tail(solution):
    assert ed._arithmetic_step(solution) is None


def test_extractor_does_not_teach_question_numbers_as_givens():
    question = '# Problem 5.\nIn 10 minutes Zhenya eats 5 cakes and Sasha eats 3. How many cakes remain from 35?'
    response = ed._extract_facts(question)['response']
    assert '# Problem' not in response
    assert '- 5\n' not in response
    assert 'Explicit numeric quantities in the givens: 10, 5, 3' in response
    # The target's 35 is never mislabeled as an observed initial condition.
    assert 'Requested target:\n- How many cakes remain from 35?' in response


def test_extractor_preserves_display_math_symbols_and_post_goal_constraints():
    question = ('Problem 5. From digits $a, b, c$, form a number. It satisfies\n\n'
                '$$\na+b+c=12\n$$\n\nFind the number. Multi-digit numbers cannot start with zero.')
    response = ed._extract_facts(question)['response']
    assert '- $$\na+b+c=12\n$$' in response
    assert 'Single-letter symbols appearing in the givens: a, b, c' in response
    facts, target = response.split('Requested target:')
    assert 'Multi-digit numbers cannot start with zero' in facts
    assert 'Multi-digit numbers cannot start with zero' not in target
    assert '- $$\n- ' not in response


def test_extractor_separates_multiple_requests_and_additional_conditions():
    question = ('4. Chords have length 8 and x=2.\n\na) Find AP.\n\n'
                'b) Suppose additionally the radius is 5. Find PT and the area.')
    response = ed._extract_facts(question)['response']
    facts, target = response.split('Requested target:')
    assert 'Suppose additionally the radius is 5' in facts
    assert '- a)' not in response and '- b)' not in response
    assert '- Find AP' in target and '- Find PT and the area' in target
    assert 'Explicit numeric quantities in the givens: 8, 2, 5' in response


def test_repeated_verifier_candidates_are_audited_without_changing_question_split(tmp_path, raw_source):
    out, _, manifest = build(tmp_path, raw_source)
    audit = manifest['quality_audit']['verifier_candidate_audit']
    # Each synthetic source question has the same local 2+3=5 step. This must
    # be disclosed, not confused with eight independent arithmetic test tasks.
    assert audit['unique_counts'] == {'train': 2, 'dev': 2, 'all': 2}
    assert audit['cross_split_overlap_count'] == 2
    assert audit['dev_rows_with_train_candidate'] == 6
    assert 'not independent proof-generalization evidence' in audit['interpretation']
    train = {r['question_hash'] for r in read_jsonl(out / 'verifier/train.jsonl')}
    dev = {r['question_hash'] for r in read_jsonl(out / 'verifier/dev.jsonl')}
    assert len(train) == 5 and len(dev) == 3 and not train & dev
