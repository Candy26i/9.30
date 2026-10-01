"""Frozen per-benchmark registry for MCQ RSI (MedQA, MMLU-Pro, GPQA, AQuA-RAT).

Every remote artifact is pinned to an exact HF revision and a content digest:
``sha256`` for LFS files and extracted tarball members, ``git_oid`` (git blob
sha1, what the HF tree API reports) for small non-LFS files.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from ..marginal_value import ADVISOR_KINDS
from ..prompt import build_manager_system_prompt

PACKAGE_ROOT = Path(__file__).resolve().parents[3]  # agent_routing/
BASE_MODEL = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ASSETS_REPO = "MaliDDD/agent-routing-9b-assets"
ASSETS_REVISION = "e240a6a8b934637418d925aa78efba431fc1fc74"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
TOKENIZER_FILES = (
    ("chat_template.jinja", "git_oid", "a585dec894e63da457d9440ec6aa7caa16d20860"),
    ("tokenizer.json", "sha256", "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    ("tokenizer_config.json", "git_oid", "4b2461e1a68b7b173cb06a81a62fb25f63e8b873"),
)
IMPORTABLE_FILES = frozenset(ADAPTER_FILES) | {name for name, _, _ in TOKENIZER_FILES} | {"special_tokens_map.json"}
ARMS = ("dynamic", "static", "success")


@dataclass(frozen=True)
class HFFile:
    repo_id: str
    repo_type: str
    revision: str
    path: str
    sha256: str = ""
    git_oid: str = ""

    @property
    def uri(self) -> str:
        prefix = "datasets/" if self.repo_type == "dataset" else ""
        return f"hf://{prefix}{self.repo_id}@{self.revision}/{self.path}"


@dataclass(frozen=True)
class TarMember:
    archive: HFFile
    member: str
    sha256: str
    rows: int = 0

    @property
    def uri(self) -> str:
        return f"{self.archive.uri}#{self.member}"


Source = Union[HFFile, TarMember]


@dataclass(frozen=True)
class Adapter:
    repo_id: str
    repo_type: str
    revision: str
    subfolder: str
    files: Tuple[Tuple[str, str, str], ...]  # (name, "sha256"|"git_oid", digest)

    def hf_files(self) -> Tuple[HFFile, ...]:
        out = []
        for name, kind, digest in self.files:
            path = f"{self.subfolder}/{name}" if self.subfolder else name
            out.append(HFFile(self.repo_id, self.repo_type, self.revision, path, **{kind: digest}))
        return tuple(out)

    @property
    def uri(self) -> str:
        prefix = "datasets/" if self.repo_type == "dataset" else ""
        return f"hf://{prefix}{self.repo_id}@{self.revision}/{self.subfolder}".rstrip("/")


@dataclass(frozen=True)
class Benchmark:
    name: str
    choice_keys: Tuple[str, ...]
    task_description: str
    manager_system_sha256: str
    cache: str
    cache_sha256: str
    cache_rows: int
    rho: float
    advisor_repo: str
    advisor_revision: str
    advisors: Tuple[Tuple[str, Adapter], ...]
    advisor_sft: Tuple[Tuple[str, TarMember], ...]
    round1_adapter: Adapter
    round1_labels: Source
    round1_records: Source
    split_manifest: str
    paper_targets: Tuple[Tuple[str, float], ...]
    depth: int = 2
    status: str = "ready"
    extra_sources: Tuple[Tuple[str, Source], ...] = ()
    aux_caches: Tuple[Tuple[str, str, str], ...] = ()  # (role, repo-relative path, sha256)
    cache_build: str = ""
    cache_sources: Tuple[Tuple[str, HFFile], ...] = ()  # (split, pinned raw file) the untracked cache is built from
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def advisor(self, kind: str) -> Adapter:
        return dict(self.advisors)[kind]

    def lora_name(self, kind: str) -> str:
        if kind not in ADVISOR_KINDS:
            raise ValueError(f"Unknown advisor kind: {kind}")
        return f"{self.name}_{kind}"

    def manager_system_prompt(self, choice_keys) -> str:
        """The paper manager system prompt lists the row's own keys (MMLU-Pro has 3-10 options)."""
        keys = list(choice_keys)
        if len(keys) < 2 or keys != list(self.choice_keys[:len(keys)]):
            raise ValueError(f"{self.name}: choice keys {keys} are not a prefix of {list(self.choice_keys)}")
        return build_manager_system_prompt(keys, self.task_description)

    def path(self, relative: str) -> Path:
        return PACKAGE_ROOT / relative


def _asset(path: str, sha256: str = "", git_oid: str = "") -> HFFile:
    return HFFile(ASSETS_REPO, "dataset", ASSETS_REVISION, path, sha256=sha256, git_oid=git_oid)


ASSETS_0814 = _asset("snapshots/assets_0814.tgz", "13ee552afb727d02b2e9e9da2813082ad1179daecb42f9b44a2a34f67710b83c")
AQUA_0817 = _asset("snapshots/aqua_arc_rescue_0817.tgz", "c45eee483f0f9fede72a941e8428892c2feef1ccd691a8e04e452c73cebff9a1")
SMALL_ASSETS = _asset("small_assets.tgz", "c08c82863b99765e6274c9e5ac8c25b40c6231f8cda678e5fd438a78b9b161ac")


def _adapter(repo_id, revision, subfolder, config_oid, model_sha, repo_type="model", tokenizer=True) -> Adapter:
    files = (("adapter_config.json", "git_oid", config_oid), ("adapter_model.safetensors", "sha256", model_sha))
    return Adapter(repo_id, repo_type, revision, subfolder, files + (TOKENIZER_FILES if tokenizer else ()))


def _advisors(repo, revision, digests, no_tokenizer=()) -> Tuple[Tuple[str, Adapter], ...]:
    return tuple(
        (kind, _adapter(repo, revision, f"{kind}_adapter", cfg, model, tokenizer=kind not in no_tokenizer))
        for kind, (cfg, model) in zip(ADVISOR_KINDS, digests)
    )


def _sft(archive: HFFile, folder: str, digests, files=None) -> Tuple[Tuple[str, TarMember], ...]:
    files = files or {}
    return tuple(
        (kind, TarMember(archive, f"outputs/sft_data/{folder}/{files.get(kind, kind + '_runtime_raw_sft.jsonl')}", digest))
        for kind, digest in zip(ADVISOR_KINDS, digests)
    )


def _mv(archive: HFFile, run: str, name: str, sha256: str, rows: int) -> TarMember:
    return TarMember(archive, f"outputs/manager/{run}/marginal_value/{name}", sha256, rows)


MEDQA = Benchmark(
    name="medqa",
    choice_keys=tuple("ABCD"),
    task_description="You are a manager agent solving a medical multiple-choice question.",
    manager_system_sha256="58f48e432a1d63b132e9c58fb5b00c39628b49e375949749493839f3cdf87c88",
    cache="outputs/data/medqa_us4_normalized.jsonl",
    cache_sha256="d5f81875589a05a7289e53dd78f277a735a38313d53391496f729b79bd342d2c",
    cache_rows=12723,
    rho=3.0,
    advisor_repo="MaliDDD/agent-routing-advisors-medqa-9b",
    advisor_revision="6bd9c4172da3cf056ab573b0a5e9eea6d4de89ed",
    advisors=_advisors(
        "MaliDDD/agent-routing-advisors-medqa-9b", "6bd9c4172da3cf056ab573b0a5e9eea6d4de89ed",
        (("21d19d6ddf5a32666d350b536da6ab27d4db062e", "981622faf7b2aa0300f51190a2468856d9b8119dd2cf30fd14d93392a6c47827"),
         ("f1c86fc10dfe19a6db4a51b53d927c50cbd05b8d", "55130cbb61d3a553de0c78c389dad6f6c43834d680f18e92190647cbb78955d9"),
         ("dc68a76c8e666712bc26d1c9653cf5ec76b328c0", "8905037c36223361ac4acff9972ac7ca7f528fa86bd015ef17ba6b5ea98417ee")),
        no_tokenizer=("verifier",),
    ),
    advisor_sft=_sft(SMALL_ASSETS, "ds_medqa", (
        "393b057a3ebf699de8cc1adf1d40ec78e0505997c19756ddf87e7dec57169fc8",
        "ee32f6958026a9fb18d6cf56b59db39d09b27a3f30624db7109ad0982c1bc9b1",
        "e72085a62e2bb4c15110671fa36891fbd4fa10855e70ba3b0e5913ee0728cc79",
    )),
    round1_adapter=_adapter(
        ASSETS_REPO, ASSETS_REVISION, "managers/medqa_marginal_9b_d2_400/sft_3.0",
        "7e9dd4125262e4e9a4ba85425b17bd49af039226", "e8ca757d0e2d608e13b1366d22fcf0d487ee7603214f3e6926f6ad01ed116340",
        repo_type="dataset",
    ),
    round1_labels=_mv(ASSETS_0814, "medqa_marginal_9b_d2_400", "manager_sft_marginal_ratio3.0.jsonl",
                      "5107102e531769135638c2d9709cee1bb6b959956724b55d46af7c7a0df0aa45", 315),
    round1_records=_mv(ASSETS_0814, "medqa_marginal_9b_d2_400", "counterfactual_records.jsonl",
                       "a780fd21571e251c8778b9ddc8e682f70bf3ca5c2cf555a5319413d9ab2eb8ab", 400),
    aux_caches=(("dev", "outputs/data/medqa_dev200.jsonl",
                 "498db4e0ac539408d092f44c68ac60788315f1c179aa741822743c86aea4c853"),),
    split_manifest="data/mcq_rsi/medqa_splits.json",
    paper_targets=(("dev_accuracy", 0.82), ("dev_avg_tool_calls", 0.405), ("dev_call_gap", 0.4738)),
    notes=(
        "S_1 substitutes d2_400/sft_3.0 (seed 0) for the locked sft_3.0_s1, which is not on HF.",
        "Verifier advisor top level is checkpoint-600 of 750 and ships no tokenizer/template files.",
    ),
)

MMLU_PRO = Benchmark(
    name="mmlu_pro",
    choice_keys=tuple("ABCDEFGHIJ"),
    task_description=(
        "You are a manager agent solving multiple-choice questions across diverse academic subjects. "
        "Each question has 10 options (A-J)."
    ),
    manager_system_sha256="14b3ccabc87f3836e00da67e2335868eb8343e6fbb50a8cac7e629ed4ae680fc",
    cache="outputs/data/mmlu_pro_normalized.jsonl",
    cache_sha256="273747caa3448dd2b3051485adb6fd59505700a0015eea6971ff7fd23fc3e23f",
    cache_rows=12032,
    rho=2.0,
    advisor_repo="MaliDDD/agent-routing-advisors-mmlu_pro-9b",
    advisor_revision="45c4c4c891541511b10a0cd4d5665e9cd8d67bdc",
    advisors=_advisors(
        "MaliDDD/agent-routing-advisors-mmlu_pro-9b", "45c4c4c891541511b10a0cd4d5665e9cd8d67bdc",
        (("0fa5209517c2fe323bbfd59223098386776db674", "06308c12f12ca6d9722fac3152b423ebfa83be855ef1b4b59997a1d3753c14db"),
         ("7c2c8eb64dbf9ac70f074a473494084f2748ab31", "09c6d47687ace2a9ce0964f6488b2e23cdb928cbbe8476578459edba457e5b50"),
         ("3537042502ea45a60d5a05ca98cc3b6e15c31721", "177c7b1dd93e5c28a62cd6d91604518d75fed11df7842fddf779d5623cd7ce81")),
    ),
    advisor_sft=_sft(SMALL_ASSETS, "ds_mmlu_pro", (
        "960485358af76f79d686981b0e87ebece34eb758d08272d55f1a1016020c5665",
        "7270ebed5bf4e456fc325d8ef27699419acd3fc04be9dde51b896101223f34b5",
        "f859fc698b9a34cae74216f8f1cc4cee73a628769a0335bd9187e552a1dfa420",
    ), files={"verifier": "verifier_runtime_raw_sft_filtered.jsonl"}),
    round1_adapter=_adapter(
        ASSETS_REPO, ASSETS_REVISION, "managers/mmlu_marginal_9b_d2/sft_2.0",
        "eb3e5fa287edc7cd68116cbd2d7f7fc9dbe1deaa", "72f44718b09cf64e26895b7ceb9cb306d0d54755cb5f495ffa073a839a78eb1f",
        repo_type="dataset",
    ),
    round1_labels=_mv(ASSETS_0814, "mmlu_marginal_9b_d2", "manager_sft_marginal_ratio2.0.jsonl",
                      "6daa14a2d7163850b858b7a98a51eab74c0951cdcb1d1df93f236a0d6bdaa78c", 331),
    round1_records=_mv(ASSETS_0814, "mmlu_marginal_9b_d2", "counterfactual_records.jsonl",
                       "ece626288b02914164f4d9ee8e55bf8ef6801c9453b3901e16ebe3997a92daa9", 400),
    split_manifest="data/mcq_rsi/mmlu_pro_splits.json",
    paper_targets=(("dev_accuracy", 0.645), ("dev_avg_tool_calls", 0.36), ("dev_call_rate", 0.34), ("dev_call_gap", 0.2248)),
    notes=(
        "S_1 is the assets depth-2 sft_2.0, not the depth-1 HF model repo mmlu_marginal_9b-sft_2.0.",
        "Choice count varies (3-10); the task description says 10 options as in the paper runs, but the "
        "DRAFT_ANSWER_/ANSWER_ key list is the row's own (6 distinct system prompts in round 1).",
        "Verifier advisor was trained on verifier_runtime_raw_sft_filtered.jsonl (598 of 600 rows).",
    ),
)

GPQA = Benchmark(
    name="gpqa",
    choice_keys=tuple("ABCD"),
    task_description="You are a manager agent solving expert-level graduate science multiple-choice questions.",
    manager_system_sha256="0922af3d86f952ef33c57e1c9125cfb1a9d7bcb4cca29a0a9ea5883a0b713678",
    cache="outputs/data/gpqa_train446.jsonl",
    cache_sha256="cf1044b8472b0a35883933640086dc41c91f87f3e88eff852eefb72aae4b8243",
    cache_rows=446,
    rho=2.0,
    advisor_repo="MaliDDD/agent-routing-advisors-gpqa-9b",
    advisor_revision="fd9ae0b4bd369fb8008c8f0bef6eea2a40e8ecff",
    advisors=_advisors(
        "MaliDDD/agent-routing-advisors-gpqa-9b", "fd9ae0b4bd369fb8008c8f0bef6eea2a40e8ecff",
        (("33bcf853eae78d0fd124a95da1be79d7b389e11a", "bac78601d5762371c88d7aba84758ea38c57219ce6d472e9b7df48ad11dc350e"),
         ("d6f6739b202db9b8756981409774175d44771d09", "a72b2c9234bfce81b587e99f0d752277f054deea5fe556b023dfdd4a07345b16"),
         ("be72638e4e0c97ce1aa56f5e119e127a21c361d2", "3b9ba504fca44ede9d92d291f3e59765e279150242da08ad59596cfb982e053a")),
    ),
    advisor_sft=_sft(SMALL_ASSETS, "ds_gpqa", (
        "b02ae80e3546fe20ec5a8223dc54932e46e0f5f5efe2f5510b9da1c5b545978a",
        "a6d4db18a54ff0cdf2a4f81f1602aff1efad2b94a440f7850c6d93e4f3f7f398",
        "a255be59d870a2e02f821309b3fa9e028df9b48171f686140a8d8e23ba5ba75a",
    )),
    round1_adapter=_adapter(
        "MaliDDD/gpqa_marginal_9b-sft_2.0", "2d10842b161a50bb5ca4529fe193c1a36332c91e", "",
        "79fef262ea567eac90709915dea9ad4f9b85651e", "cd889041acb6d05ba3d4a41faa70f7d94f2fb05d438baa9acc1a629fd1d99407",
    ),
    round1_labels=HFFile("MaliDDD/marginal-gpqa_marginal_9b", "dataset", "039838e7cd7f6d8a6423c762412c7fce55aa18b1",
                         "manager_sft_marginal_ratio2.0.jsonl",
                         sha256="845df601fbedd7043645c243742132c46af46efeb9cb0903b2ba860178f594a7",
                         git_oid="494d71be45a9881e0526fe73ae6dbf7c1548df5e"),
    round1_records=HFFile("MaliDDD/marginal-gpqa_marginal_9b_d2", "dataset", "04a2152827d196c21e1350e4042fe2bd83699b29",
                          "counterfactual_records.jsonl",
                          sha256="3027ed1418bc92c970de3a07ba732c76a6eecc0b35cd135eceec57afa9e654c8",
                          git_oid="99488bb136064c1dc11adbca1272f331689847a8"),
    extra_sources=(
        ("label_records", HFFile("MaliDDD/marginal-gpqa_marginal_9b", "dataset", "039838e7cd7f6d8a6423c762412c7fce55aa18b1",
                                 "counterfactual_records.jsonl",
                                 sha256="c5d9c8a7ec7c2256ebeba0406bde43e12b94fd0f6994b60ad534bf2becea5130",
                                 git_oid="57cda8dcc848019649023ed87f4ac7d5298b6f7a")),
    ),
    aux_caches=(("test", "outputs/data/gpqa_diamond_eval100.jsonl",
                 "c2463316911def54735807d4daf1d37fff91f084b113a3d4ff570b8a543d2607"),),
    split_manifest="data/mcq_rsi/gpqa_splits.json",
    paper_targets=(("test_accuracy", 0.54), ("test_avg_tool_calls", 0.51), ("test_call_gap", 0.0172)),
    notes=(
        "S_1 (locked sft_2.0) was trained on the depth-1 collection; round1_records is the depth-2 tree on the same 200 roots.",
        "No spare data: collect_r2/r3 reuse the 200 collect_r1 roots and grpo_r1..r3 reuse one 146-row pool.",
        "Parity targets are on Diamond-100 (= test); dev is new and has no paper number.",
    ),
)

AQUA = Benchmark(
    name="aqua",
    choice_keys=tuple("ABCDE"),
    task_description="You are a manager agent solving five-choice algebraic word problems.",
    manager_system_sha256="33bb5b7ba17017f64a093bbb9f36ce49480c9baa66cbd082e56c24b22c76bd50",
    cache="outputs/data/aqua_rat_normalized.jsonl",
    cache_sha256="f6bc028029fea092ab9c12b8ae496c7b92966f326ea0df3f5fd407be5a00c9b2",
    cache_rows=97975,
    rho=1.0,
    advisor_repo="MaliDDD/agent-routing-advisors-aqua-9b",
    advisor_revision="64c76a8de739daff2f4d6a99e734d13894f042f9",
    advisors=_advisors(
        "MaliDDD/agent-routing-advisors-aqua-9b", "64c76a8de739daff2f4d6a99e734d13894f042f9",
        (("b5c34d8852cd10342a9f33b214c415314b2b531b", "11ea3fe8c63897966465ea9b22486732145a9b4e261cb9bf87a7dbaf7d027b34"),
         ("e34fa48e0afafbc5f03ca3bd5acd9e214874f678", "52c375b7efb9df9ac78a9cf170dbac8f1d1c733f6029b733182e4f2a525dd676"),
         ("899bd45f27726c82de2aa213febee56b9e394048", "8fefacec561320c4016887273a6d03c27413416cdf588a8a67916de4fed67ac6")),
    ),
    advisor_sft=_sft(AQUA_0817, "ds_aqua_9B_sub", (
        "f1d7d6c5f09818dc9312ae6ebb824771a1173c93cb4a244f59d941bbbec67e11",
        "838ff4120390f63f5c8924bf06c3c22cb80e0efce20ffe247e9969faf4d36f74",
        "e7ff1ea0374700d4f7a56532ee9628bda1bd668eff06df6eabdc64adb0bcc2bc",
    )),
    round1_adapter=_adapter(
        ASSETS_REPO, ASSETS_REVISION, "managers/aqua_marginal_9b_v1/sft_1.0",
        "a55a729968ed60cc0319afdc7074cb049e19204a", "a25318ad635cba23b5c8fddb69d25d294eb868a6e9a1c63ab6dae4604697d4e1",
        repo_type="dataset",
    ),
    round1_labels=_mv(AQUA_0817, "aqua_marginal_9b_v1", "manager_sft_marginal_ratio1.0.jsonl",
                      "89aa9e00c45b20f731428f4416bae3764d4394e7c9608efd7d7d8949f6be5753", 402),
    round1_records=_mv(AQUA_0817, "aqua_marginal_9b_v1", "counterfactual_records.jsonl",
                       "160634f4c5b6ac3737c3b159a4ede3a267d0698dd2fd83b6ebda9de860aeac71", 400),
    split_manifest="data/mcq_rsi/aqua_splits.json",
    paper_targets=(("dev_accuracy", 0.78), ("dev_paper_calls", 0.894), ("test_accuracy", 0.768), ("test_paper_calls", 0.78)),
    extra_sources=(
        ("round1_labels_alt", _mv(AQUA_0817, "aqua_marginal_9b_v1", "manager_sft_marginal_d2_ratio1.0.jsonl",
                                  "2c1af5294d623888ea2da4b1e223eeebe5b76525ed0043b967a2fb1719a66c19", 402)),
    ),
    cache_build="src.benchmarks.aqua_rat.load_aqua_rat(source='local') over cache_sources (deepmind/aqua_rat raw parquet)",
    cache_sources=tuple(
        (split, HFFile("deepmind/aqua_rat", "dataset", "33301c6a050c96af81f63cad5562cb5363e88971",
                       f"raw/{split}-00000-of-00001.parquet", sha256=sha))
        for split, sha in (("train", "b9e0ee4d72661e3d00002553fb210938f50528b42cc72809fde18b8119c96778"),
                           ("validation", "99605f7d8e5b322445e6585cf6f1d0144f7e8e6f1e2f99672d40dc53fd3e61f5"),
                           ("test", "3bb85756cac23fe0b0aba0b9fd0567a73f075d5aa5511b8bd7499d1cd8dbb4d1"))
    ),
    notes=(
        "The normalized cache is not tracked; prepare-splits rebuilds it from cache_sources with the 9.30 loader "
        "and checks cache_sha256.",
        "Paper targets come from the paper text only; no AQuA eval record is on HF.",
        "*_paper_calls are the paper 'Calls' columns (Tables 3/4), probably tool_call_rate, not avg_tool_calls: "
        "Table 4's MMLU-Pro 0.340 is the eval log's tool_call_rate (avg 0.36), and Table 9 lists the 81.1% AQuA "
        "rho=0 point at 1.000 calls where Table 5 gives 1.031. Parity gates accept either metric until S_1 is re-evaluated.",
        "round1_labels is the seed-0 _balance_records(ratio 1.0) output (reproduced byte for byte). The same "
        "directory holds manager_sft_marginal_d2_ratio1.0.jsonl (round1_labels_alt, sha256 2c1af529...), same "
        "decision counts but a different 52 commit rows, from the ad hoc depth-cut script and no seed in 0-99. "
        "Confirm with S_1's loss on those rows before the static arm uses round1_labels.",
        "All 400 collect_r1 roots are advisor-SFT questions.",
    ),
)

BENCHMARKS: Dict[str, Benchmark] = {b.name: b for b in (MEDQA, MMLU_PRO, GPQA, AQUA)}


def get(name: str) -> Benchmark:
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise KeyError(f"Unknown MCQ benchmark {name!r}; choose from {sorted(BENCHMARKS)}") from None


def _sources(bench: Benchmark):
    yield "round1_labels", bench.round1_labels
    yield "round1_records", bench.round1_records
    for kind, member in bench.advisor_sft:
        yield f"advisor_sft/{kind}", member
    yield from bench.extra_sources


def validate(bench: Optional[Benchmark] = None) -> None:
    """Internal-consistency checks; raises ValueError on the first problem."""
    benches = [bench] if bench is not None else list(BENCHMARKS.values())
    hex64 = lambda s: len(s) == 64 and all(c in "0123456789abcdef" for c in s)
    hex40 = lambda s: len(s) == 40 and all(c in "0123456789abcdef" for c in s)
    lora_names = set()
    for b in benches:
        def fail(msg):
            raise ValueError(f"{b.name}: {msg}")
        if BENCHMARKS.get(b.name) is not b and bench is None:
            fail("registry key mismatch")
        if list(b.choice_keys) != [chr(ord("A") + i) for i in range(len(b.choice_keys))]:
            fail("choice keys must be contiguous from A")
        sha = hashlib.sha256(b.manager_system_prompt(b.choice_keys).encode("utf-8")).hexdigest()
        if sha != b.manager_system_sha256:
            fail("task description does not reproduce the recorded manager system prompt")
        if b.depth != 2 or b.rho <= 0:
            fail("depth must be 2 and rho positive")
        if b.split_manifest != f"data/mcq_rsi/{b.name}_splits.json":
            fail("split manifest path")
        if b.status not in {"ready", "pending"} or not hex64(b.cache_sha256):
            fail("status/cache digest")
        if [k for k, _ in b.advisors] != list(ADVISOR_KINDS) or [k for k, _ in b.advisor_sft] != list(ADVISOR_KINDS):
            fail("advisor kinds")
        for adapter in [a for _, a in b.advisors] + [b.round1_adapter]:
            if not hex40(adapter.revision) or adapter.repo_type not in {"model", "dataset"}:
                fail(f"unpinned adapter {adapter.uri}")
            names = [n for n, _, _ in adapter.files]
            if names[:2] != list(ADAPTER_FILES) or len(set(names)) != len(names) or set(names) - IMPORTABLE_FILES:
                fail(f"adapter files {adapter.uri}")
            if "checkpoint-" in adapter.subfolder:
                fail("checkpoint subfolders are never imported")
            for name, kind, digest in adapter.files:
                if not (hex64(digest) if kind == "sha256" else hex40(digest)):
                    fail(f"digest {adapter.uri}/{name}")
        for kind, adapter in b.advisors:
            if adapter.repo_id != b.advisor_repo or adapter.revision != b.advisor_revision:
                fail(f"{kind} advisor repo/revision")
            if adapter.subfolder != f"{kind}_adapter":
                fail(f"{kind} advisor subfolder")
            name = b.lora_name(kind)
            if name in lora_names:
                fail(f"duplicate LoRA name {name}")
            lora_names.add(name)
        for label, src in _sources(b):
            archive = src.archive if isinstance(src, TarMember) else src
            if not hex64(src.sha256) or not hex40(archive.revision):
                fail(f"unpinned source {label}")
            if isinstance(src, TarMember) and not hex64(archive.sha256):
                fail(f"unpinned archive {archive.path}")
        for role, path, digest in b.aux_caches:
            if role not in {"dev", "test"} or not hex64(digest) or path.startswith("/"):
                fail(f"aux cache {path}")
        for split, src in b.cache_sources:
            if split not in {"train", "validation", "test"} or not hex64(src.sha256) or not hex40(src.revision):
                fail(f"unpinned cache source {split}")
