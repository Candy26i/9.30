import json

import pytest

from src.benchmarks.base import StandardRow
from src.verifiable.expert_data import NEAR_THRESHOLD
from src.verifiable.expert_isolation import verify_manager_rows
from src.verifiable.expert_train import digest
from test_expert_train import make_data, rewrite_rows, write_json
from test_expert_serving import write_bundle
from src.verifiable.runner import checkpoint_identity


def config_for(data):
    return {'expert_data_manifest': str(data / 'manifest.json'),
            'expert_data_manifest_sha256': digest(data / 'manifest.json')}


def test_plain_manager_configs_remain_unchanged_and_incomplete_binding_fails():
    assert verify_manager_rows({}, [{}])['checked'] is False
    with pytest.raises(ValueError, match='without'):
        verify_manager_rows({'expert_data_manifest_sha256': 'a' * 64}, [])


@pytest.mark.parametrize('split', ['train', 'dev'])
def test_exact_expert_overlap_rejected_for_standard_rows_and_hash_only_sft(tmp_path, split):
    data = make_data(tmp_path / 'data')
    row = json.loads((data / 'verifier' / f'{split}.jsonl').read_text().splitlines()[0])
    standard = StandardRow(0, 'manager', 'free_response_math', row['question'], {}, '2', split='train')
    cfg = config_for(data)
    for manager_row in (standard, {'question_hash': row['question_hash'], 'prompt': [], 'response': []}):
        with pytest.raises(ValueError, match='expert train/dev pool'):
            verify_manager_rows(cfg, [manager_row])


def test_near_overlap_uses_builder_index_and_nonoverlap_reports_hash_only_limit(tmp_path):
    from src.verifiable.data import identity
    from src.verifiable.protocol import advisor_messages
    from types import SimpleNamespace
    data = make_data(tmp_path / 'data')
    question = ' '.join(f'variable{i}' for i in range(60)) + ' Find the sum.'
    def replace(rows):
        rows[0]['question'] = question
        rows[0]['question_hash'] = identity(question)
        rows[0]['prompt'] = advisor_messages('reasoner', SimpleNamespace(question=question, context=''), '')
    rewrite_rows(data, 'reasoner/train.jsonl', replace)
    cfg = config_for(data)
    with pytest.raises(ValueError, match=r'pool \(near\)'):
        verify_manager_rows(cfg, [{'question': question.replace('Find the sum', 'Please find the sum')}])
    rows = [{'question': 'A completely unrelated integration problem about a circle and its radius.'},
            {'question_hash': identity('Historical SFT row without a copied question field')}]
    result = verify_manager_rows(cfg, rows)
    assert result['checked_rows'] == 2 and result['text_rows_with_near_check'] == result['exact_only_rows'] == 1
    assert result['near_jaccard_threshold'] == NEAR_THRESHOLD


def test_frozen_manifest_file_data_and_supplied_question_hash_are_verified(tmp_path):
    data = make_data(tmp_path / 'data')
    cfg = config_for(data)
    with pytest.raises(ValueError, match='disagree'):
        verify_manager_rows(cfg, [{'question': 'Different problem', 'question_hash': '0' * 64}])
    with pytest.raises(ValueError, match='without question'):
        verify_manager_rows(cfg, [{'prompt': [], 'response': []}])
    original = (data / 'manifest.json').read_bytes()
    (data / 'manifest.json').write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='manifest fingerprint'):
        verify_manager_rows(cfg, [])
    (data / 'manifest.json').write_bytes(original)
    path = data / 'reasoner/dev.jsonl'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='data fingerprint'):
        verify_manager_rows(cfg, [])


def test_unrelated_valid_manifest_cannot_be_attached_to_frozen_adapters(tmp_path):
    data = make_data(tmp_path / 'data')
    bundle_root = tmp_path / 'bundle'
    bundle_root.mkdir()
    path, value = write_bundle(bundle_root)
    for role in value['roles']:
        directory = bundle_root / role
        write_json(directory / 'summary.json', {'training_complete': True, 'role': role,
            'base_model': value['base_model'], 'base_model_revision': value['base_model_revision'],
            'template_sha256': value['template_sha256'], 'data_manifest_sha256': '0' * 64})
        value['roles'][role]['identity'] = checkpoint_identity(str(directory))
    write_json(path, value)
    cfg = {**config_for(data), 'advisor_expert_bundle': str(path)}
    with pytest.raises(ValueError, match='does not bind'):
        verify_manager_rows(cfg, [])

    for role in value['roles']:
        directory = bundle_root / role
        summary = json.loads((directory / 'summary.json').read_text())
        summary['data_manifest_sha256'] = cfg['expert_data_manifest_sha256']
        write_json(directory / 'summary.json', summary)
        value['roles'][role]['identity'] = checkpoint_identity(str(directory))
    write_json(path, value)
    assert verify_manager_rows(cfg, [{'question': 'Separate geometry question about an ellipse.'}])['checked']
