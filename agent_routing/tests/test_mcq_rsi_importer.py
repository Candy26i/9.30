import dataclasses
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from src.manager.marginal_value import ADVISOR_KINDS
from src.manager.mcq_rsi import __main__ as cli
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi import importer
from src.manager.mcq_rsi.benchmarks import BENCHMARKS, HFFile, TarMember

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mcq_rsi" / "adapter_configs"


def _no_network(**kwargs):
    raise AssertionError(f"network access attempted: {kwargs}")


def test_plan_only_names_top_level_adapter_files():
    for b in BENCHMARKS.values():
        items = importer.plan(b)
        dests = [i["dest"] for i in items]
        assert len(dests) == len(set(dests))
        for item in items:
            src = item["source"]
            path = src.member if isinstance(src, TarMember) else src.path
            assert "checkpoint-" not in path
            if item["part"] in {"advisors", "manager"}:
                assert path.rsplit("/", 1)[-1] in registry.IMPORTABLE_FILES
        assert {d.split("/")[1] for d in dests if d.startswith("advisors/")} == set(ADVISOR_KINDS)
        assert "round1/records.jsonl" in dests and "round1/labels.jsonl" in dests
    medqa_verifier = [i for i in importer.plan(BENCHMARKS["medqa"], ("advisors",)) if i["dest"].startswith("advisors/verifier/")]
    assert [i["dest"] for i in medqa_verifier] == ["advisors/verifier/adapter_config.json", "advisors/verifier/adapter_model.safetensors"]


def test_dry_run_never_downloads(tmp_path, capsys, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_network)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    result = importer.run_import(BENCHMARKS["gpqa"], str(tmp_path), dry_run=True, downloader=_no_network)
    assert result["dry_run"] and len(result["files"]) == len(importer.plan(BENCHMARKS["gpqa"]))
    assert not any(tmp_path.iterdir())
    assert cli.main(["import", "--bench", "medqa", "--dry-run", "--out", str(tmp_path)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert list(listing) == ["medqa"]
    assert all("checkpoint-" not in f["uri"] for f in listing["medqa"]["files"])
    assert cli.main(["import", "--dry-run", "--out", str(tmp_path)]) == 0
    assert list(json.loads(capsys.readouterr().out)) == list(BENCHMARKS)
    assert not any(tmp_path.iterdir())
    with pytest.raises(SystemExit):
        cli.main(["import", "--dry-run", "--parts", "", "--out", str(tmp_path)])
    with pytest.raises(ValueError):
        importer.plan(BENCHMARKS["gpqa"], ())


def _fake_bench(tmp_path):
    """MedQA registry entry re-pointed at local fake content with real digests."""
    store = {}

    def put(repo_id, path, data):
        store[(repo_id, path)] = data
        return hashlib.sha256(data).hexdigest(), importer.git_blob_oid(data)

    config = (FIXTURES / "medqa" / "s1" / "adapter_config.json").read_bytes()

    def adapter(src):
        files = []
        for name, kind, _ in src.files:
            path = f"{src.subfolder}/{name}" if src.subfolder else name
            data = config if name == "adapter_config.json" else f"{src.repo_id}/{path}".encode()
            sha, oid = put(src.repo_id, path, data)
            files.append((name, kind, sha if kind == "sha256" else oid))
        return dataclasses.replace(src, files=tuple(files))

    b = BENCHMARKS["medqa"]
    members = {"labels": b"label\n", "records": b"record\n", **{k: f"{k} sft\n".encode() for k in ADVISOR_KINDS}}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        names = {"labels": b.round1_labels.member, "records": b.round1_records.member,
                 **{k: m.member for k, m in b.advisor_sft}}
        for key, data in members.items():
            info = tarfile.TarInfo(names[key])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar.addfile(tarfile.TarInfo("outputs/unrelated.txt"), io.BytesIO(b""))
    archive_sha, _ = put(registry.ASSETS_REPO, "fake.tgz", buf.getvalue())
    archive = HFFile(registry.ASSETS_REPO, "dataset", registry.ASSETS_REVISION, "fake.tgz", sha256=archive_sha)
    member = lambda m, key: TarMember(archive, m.member, hashlib.sha256(members[key]).hexdigest())
    fake = dataclasses.replace(
        b,
        advisors=tuple((k, adapter(a)) for k, a in b.advisors),
        round1_adapter=adapter(b.round1_adapter),
        round1_labels=member(b.round1_labels, "labels"),
        round1_records=member(b.round1_records, "records"),
        advisor_sft=tuple((k, member(m, k)) for k, m in b.advisor_sft),
    )
    calls = []

    def downloader(repo_id, filename, repo_type, revision, cache_dir=None):
        assert revision in {registry.ASSETS_REVISION, b.advisor_revision} and "checkpoint-" not in filename
        calls.append(filename)
        target = tmp_path / "hub" / repo_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(store[(repo_id, filename)])
        return str(target)

    return fake, downloader, calls


def test_mocked_import_verifies_and_writes_manifest(tmp_path):
    fake, downloader, calls = _fake_bench(tmp_path)
    out = tmp_path / "import"
    manifest = importer.run_import(fake, str(out), downloader=downloader)
    root = out / "medqa"
    assert {f["status"] for f in manifest["files"]} == {"downloaded", "extracted"}
    assert calls.count("fake.tgz") == 1
    assert (root / "round1/records.jsonl").read_bytes() == b"record\n"
    assert (root / "advisor_sft/verifier.jsonl").read_bytes() == b"verifier sft\n"
    assert manifest["lora_names"] == {k: f"medqa_{k}" for k in ADVISOR_KINDS}
    on_disk = json.loads((root / "import_manifest.json").read_text())
    assert on_disk == manifest and not list(root.rglob("*.part"))
    for f in manifest["files"]:
        assert hashlib.sha256((root / f["dest"]).read_bytes()).hexdigest() == f["sha256"]

    calls.clear()
    again = importer.run_import(fake, str(out), downloader=downloader)
    assert calls == [] and {f["status"] for f in again["files"]} == {"present"}

    (root / "advisors/reasoner/adapter_model.safetensors").write_bytes(b"corrupt")
    again = importer.run_import(fake, str(out), downloader=downloader)
    assert calls == ["reasoner_adapter/adapter_model.safetensors"]


def test_mocked_import_rejects_digest_mismatch(tmp_path):
    fake, downloader, _ = _fake_bench(tmp_path)
    kind, adapter = fake.advisors[0]
    files = tuple((n, k, "0" * 64 if n == "adapter_model.safetensors" else d) for n, k, d in adapter.files)
    bad = dataclasses.replace(fake, advisors=((kind, dataclasses.replace(adapter, files=files)),) + fake.advisors[1:])
    with pytest.raises(ValueError, match="sha256"):
        importer.run_import(bad, str(tmp_path / "import"), downloader=downloader, parts=("advisors",))
    assert not (tmp_path / "import/medqa/advisors/extractor/adapter_model.safetensors").exists()
    bad_member = dataclasses.replace(fake.round1_records, sha256="1" * 64)
    with pytest.raises(ValueError):
        importer.run_import(dataclasses.replace(fake, round1_records=bad_member), str(tmp_path / "import2"),
                            downloader=downloader, parts=("data",))

    files = tuple((n, k, "0" * 40 if n == "adapter_config.json" else d) for n, k, d in adapter.files)
    bad = dataclasses.replace(fake, advisors=((kind, dataclasses.replace(adapter, files=files)),) + fake.advisors[1:])
    with pytest.raises(ValueError, match="git blob oid"):
        importer.run_import(bad, str(tmp_path / "import3"), downloader=downloader, parts=("advisors",))
    assert not list((tmp_path / "import3").rglob("adapter_config.json*"))

    archive = dataclasses.replace(fake.round1_records.archive, sha256="2" * 64)
    swap = lambda m: dataclasses.replace(m, archive=archive)
    bad = dataclasses.replace(fake, round1_labels=swap(fake.round1_labels), round1_records=swap(fake.round1_records),
                              advisor_sft=tuple((k, swap(m)) for k, m in fake.advisor_sft))
    with pytest.raises(ValueError, match="sha256"):
        importer.run_import(bad, str(tmp_path / "import4"), downloader=downloader, parts=("data",))
    assert not [p for p in (tmp_path / "import4").rglob("*") if p.is_file()]


def test_import_requires_digests(tmp_path):
    fake, downloader, _ = _fake_bench(tmp_path)
    kind, adapter = fake.advisors[0]
    files = tuple((n, k, "") for n, k, _ in adapter.files)
    bad = dataclasses.replace(fake, advisors=((kind, dataclasses.replace(adapter, files=files)),) + fake.advisors[1:])
    with pytest.raises(ValueError):
        importer.run_import(bad, str(tmp_path / "import"), downloader=downloader, parts=("advisors",))
    (tmp_path / "f").write_bytes(b"x")
    with pytest.raises(ValueError, match="no registry digest"):
        importer._verify(tmp_path / "f")


def test_manifest_carries_only_verified_entries(tmp_path):
    fake, downloader, _ = _fake_bench(tmp_path)
    out = tmp_path / "import"
    first = importer.run_import(fake, str(out), downloader=downloader)
    root = out / "medqa"
    (root / "advisors/reasoner/adapter_model.safetensors").unlink()
    (root / "advisors/verifier/adapter_model.safetensors").write_bytes(b"corrupt")
    again = importer.run_import(fake, str(out), downloader=downloader, parts=("data",))
    dests = {f["dest"] for f in again["files"]}
    assert "advisors/reasoner/adapter_model.safetensors" not in dests
    assert "advisors/verifier/adapter_model.safetensors" not in dests
    assert dests == {f["dest"] for f in first["files"]} - {"advisors/reasoner/adapter_model.safetensors",
                                                           "advisors/verifier/adapter_model.safetensors"}
    for f in again["files"]:
        assert hashlib.sha256((root / f["dest"]).read_bytes()).hexdigest() == f["sha256"]
    assert {f["status"] for f in again["files"] if f["part"] == "data"} == {"present"}


def test_checkpoint_paths_are_refused():
    b = BENCHMARKS["gpqa"]
    bad = dataclasses.replace(b.round1_adapter, subfolder="checkpoint-190")
    with pytest.raises(ValueError):
        importer.plan(dataclasses.replace(b, round1_adapter=bad))
    with pytest.raises(ValueError):
        importer._download(_no_network, HFFile("r", "model", "0" * 40, "x/checkpoint-1/adapter_model.safetensors"), None)


def _fixture_configs():
    return sorted(FIXTURES.glob("*/*/adapter_config.json"))


def test_vendored_configs_are_the_pinned_files():
    configs = _fixture_configs()
    assert len(configs) == 16
    for path in configs:
        bench, role = path.parts[-3], path.parts[-2]
        b = BENCHMARKS[bench]
        adapter = b.round1_adapter if role == "s1" else b.advisor(role)
        oid = dict((n, d) for n, _, d in adapter.files)["adapter_config.json"]
        assert importer.git_blob_oid(path.read_bytes()) == oid
        importer.check_adapter_config(path)


def test_adapter_configs_load_under_pinned_peft():
    peft = pytest.importorskip("peft")
    import warnings

    for path in _fixture_configs():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = peft.PeftConfig.from_pretrained(str(path.parent))
        assert cfg.base_model_name_or_path == registry.BASE_MODEL
        assert (cfg.r, cfg.lora_alpha) == (16, 32)
        assert set(cfg.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def test_check_adapter_config_rejects_other_shapes(tmp_path):
    cfg = json.loads((FIXTURES / "gpqa" / "s1" / "adapter_config.json").read_text())
    cfg["target_modules"] = cfg["target_modules"] + ["in_proj_qkv"]
    (tmp_path / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        importer.check_adapter_config(tmp_path / "adapter_config.json")


def test_show_registry_cli(capsys):
    assert cli.main(["show-registry", "--bench", "aqua"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["aqua"]["lora_names"]["reasoner"] == "aqua_reasoner"
    assert out["aqua"]["prompts"]["reasoner"]["repo_commit"] is None
