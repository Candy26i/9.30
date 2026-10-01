"""Per-benchmark advisor system prompts, pinned by sha256.

Each ``prompts/<bench>/<kind>.txt`` is the system message of that benchmark's
advisor SFT data, byte for byte. User messages are unchanged from
``build_runtime_messages``: it reproduces every user turn of all twelve advisor
SFT files (the Verifier with its ``CANDIDATE ANSWER TO AUDIT`` block when set).

Provenance per prompt:
- ``source``: the advisor SFT file the text was copied from (all twelve agree
  within their file, so ``status`` is ``sft_data`` for every prompt);
- ``repo_commit``: the repo revision whose ``runtime_prompts.py`` holds the same
  text, or ``None`` when no commit of 7.98 or 9.30 does;
- ``runtime_confirmed``: whether greedy replay of recorded tool outputs pinned
  the prompt as the one live during the paper runs (design §7.2, still pending).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from ...subagents.prompts.runtime_prompts import build_runtime_messages
from ..marginal_value import ADVISOR_KINDS
from . import benchmarks as registry

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_COMMIT_24E6902 = "24e6902"

PROMPTS: Dict[str, Dict[str, Dict[str, object]]] = {}


def _entry(bench: str, kind: str, sha256: str, repo_commit=_COMMIT_24E6902) -> None:
    member = dict(registry.get(bench).advisor_sft)[kind]
    PROMPTS.setdefault(bench, {})[kind] = {
        "sha256": sha256,
        "path": f"prompts/{bench}/{kind}.txt",
        "source": member.uri,
        "status": "sft_data",
        "repo_commit": repo_commit,
        "runtime_confirmed": False,
    }


EXTRACTOR_SHA = "8b8b7c102ff0683647bbfd4e1fe1d4936d600be694b8a93e7fb5a8ccf3f42a6b"
VERIFIER_SHA = "71c5d71fea49f47c115d5b81a6b6377e3b57d46c12026185d95efcc83658ddd4"
ACADEMIC_REASONER_SHA = "7d3a8e8c6a66c839f45cf4100d5e8b4803fc49d749e26d0c6cdd14443c2422e6"
for _bench, _reasoner, _commit in (
    ("medqa", "415c4b2969b92f418bffabd19d6e297a836fe2b1b24baee325c805034a38c232", _COMMIT_24E6902),
    ("mmlu_pro", ACADEMIC_REASONER_SHA, None),
    ("gpqa", ACADEMIC_REASONER_SHA, None),
    ("aqua", "8ad0fc271569864dc5317f616a7584360342e9e4c8be18f6b3ae21edb79a3658", None),
):
    _entry(_bench, "extractor", EXTRACTOR_SHA)
    _entry(_bench, "reasoner", _reasoner, _commit)
    _entry(_bench, "verifier", VERIFIER_SHA)


def prompt_path(bench: str, kind: str) -> Path:
    if kind not in ADVISOR_KINDS:
        raise ValueError(f"Unknown advisor kind: {kind}")
    return PROMPT_DIR / registry.get(bench).name / f"{kind}.txt"


def system_prompt(bench: str, kind: str) -> str:
    data = prompt_path(bench, kind).read_bytes()
    expected = PROMPTS[bench][kind]["sha256"]
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"{bench}/{kind} prompt does not match its registered sha256")
    return data.decode("utf-8")


def prompt_sha256(bench: str, kind: str) -> str:
    return str(PROMPTS[bench][kind]["sha256"])


def build_advisor_messages(
    bench: str,
    kind: str,
    question: str,
    context: str,
    choices: Dict[str, str],
    candidate_answer: str = "",
) -> List[Dict[str, str]]:
    """``build_runtime_messages`` with the benchmark's trained system prompt."""
    messages = build_runtime_messages(kind, question, context, choices, candidate_answer=candidate_answer)
    if messages[0]["role"] != "system":
        raise ValueError("runtime messages must start with the system turn")
    return [{"role": "system", "content": system_prompt(bench, kind)}] + messages[1:]


def validate() -> None:
    for bench in registry.BENCHMARKS:
        if set(PROMPTS.get(bench, {})) != set(ADVISOR_KINDS):
            raise ValueError(f"{bench}: prompt registry must cover {ADVISOR_KINDS}")
        for kind in ADVISOR_KINDS:
            system_prompt(bench, kind)
