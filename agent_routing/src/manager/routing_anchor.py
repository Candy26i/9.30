"""Tokenization helpers for SFT-anchored manager GRPO experiments.

Two auxiliary objectives are supported without changing the manager's runtime
protocol:

``full``
    Replay the complete marginal-SFT assistant turn.  This anchors both the
    current draft and the subsequent routing/protocol behavior.

``route_only``
    Mask the prompt *and* the current ``DRAFT_ANSWER_*``.  Loss starts only at
    the suffix that realizes the routing action: a native tool call for CALL,
    or ``ANSWER_*`` for COMMIT.  The answer draft itself therefore receives no
    auxiliary SFT gradient.

The latter works because the existing manager protocol makes a draft before
deciding whether to call a tool or commit; no new ROUTE_* tokens are needed.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


ANCHOR_MODES = ("full", "route_only")
_DRAFT_RE = re.compile(r"DRAFT_ANSWER_[A-Za-z0-9_]+")


def _render_chat(tokenizer: Any, messages: List[Dict[str, Any]], add_generation_prompt: bool, tools=None) -> str:
    extra = {"tools": tools} if tools else {}
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
            **extra,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **extra,
        )


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    n = min(len(left), len(right))
    i = 0
    while i < n and left[i] == right[i]:
        i += 1
    return i


def _normalize_response(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, dict):
        return [response]
    if isinstance(response, str):
        return [{"role": "assistant", "content": response}]
    if isinstance(response, list):
        return list(response)
    raise TypeError(f"Unsupported manager SFT response type: {type(response).__name__}")


def _draft_prefix_message(response_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return an assistant message containing only the already-made draft.

    The returned message intentionally omits ``tool_calls``.  Rendering this
    prefix and comparing it with the full target locates the token boundary at
    which CALL vs COMMIT begins in the native chat template.
    """
    if len(response_messages) != 1 or response_messages[0].get("role") != "assistant":
        raise ValueError("route_only anchor expects exactly one assistant response message")
    content = str(response_messages[0].get("content") or "")
    match = _DRAFT_RE.search(content)
    if match is None:
        raise ValueError(
            "route_only anchor requires DRAFT_ANSWER_* in every target response; "
            f"got content={content[:120]!r}"
        )
    return {"role": "assistant", "content": match.group(0)}


def _boundary_after_text_with_offsets(
    tokenizer: Any,
    full_text: str,
    needle: str,
) -> Optional[int]:
    """Find a conservative token boundary after ``needle`` when offsets exist.

    Fast tokenizers can merge the last draft character with following
    whitespace/markup.  Masking every token whose span touches the draft makes
    the route-only claim stronger: no token overlapping DRAFT_ANSWER_* is ever
    supervised.  Slow/custom tokenizers fall back to rendered-prefix matching.
    """
    char_start = full_text.rfind(needle)
    if char_start < 0:
        return None
    char_end = char_start + len(needle)
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError, NotImplementedError):
        return None
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None
    for index, pair in enumerate(offsets):
        if pair is None or len(pair) != 2:
            continue
        start, end = int(pair[0]), int(pair[1])
        if start >= char_end and end > start:
            return index
    return len(offsets)


def response_text_from_render(prompt_text: str, full_text: str) -> str:
    """Keep the exact inference prefix instead of guessing a BPE boundary.

    Native Qwen3 non-thinking generation appends an empty thinking block to
    the assistant prefix, whereas its completed-turn render can omit it. That
    documented template difference is reconciled explicitly; arbitrary prompt
    rewrites are rejected because a common token prefix would label context.
    """
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text):]
    for suffix in ("<think>\n\n</think>\n\n", "<think>\n</think>\n\n"):
        if prompt_text.endswith(suffix) and full_text.startswith(prompt_text[:-len(suffix)]):
            return full_text[len(prompt_text) - len(suffix):]
    raise ValueError("Chat template rewrites the prompt; response-only masking cannot be verified")


def tokenize_anchor_row(
    row: Dict[str, Any], tokenizer: Any, max_seq_len: int, mode: str, tools=None,
) -> Tuple[Optional[Dict[str, List[int]]], Dict[str, int]]:
    """Encode inference prompt and assistant target separately, without truncation.

    Returning None for overlength/empty targets lets callers report exclusions;
    no partial solution or all--100 example is allowed into the loss.
    """
    if mode not in ANCHOR_MODES:
        raise ValueError(f"Unknown SFT anchor mode {mode!r}; expected one of {ANCHOR_MODES}")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    prompt_messages = list(row["prompt"])
    response_messages = _normalize_response(row["response"])
    if len(response_messages) != 1 or response_messages[0].get("role") != "assistant":
        raise ValueError("SFT requires exactly one assistant response; context must stay in prompt")
    prompt_text = _render_chat(tokenizer, prompt_messages, True, tools)
    full_text = _render_chat(tokenizer, prompt_messages + response_messages, False, tools)
    target_text = response_text_from_render(prompt_text, full_text)
    eos = tokenizer.eos_token or ""
    if eos and not target_text.rstrip().endswith(eos):
        target_text += eos
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    target_ids = list(tokenizer(target_text, add_special_tokens=False)["input_ids"])
    input_ids = prompt_ids + target_ids
    label_boundary = len(prompt_ids)
    if mode == "route_only":
        decision_type = str(row.get("decision_type") or "")
        if decision_type not in {"call", "commit", "commit_after_call"}:
            raise ValueError("route_only anchor requires decision_type in {call, commit, commit_after_call}")
        draft = str(_draft_prefix_message(response_messages)["content"])
        end = target_text.rfind(draft) + len(draft)
        if end < len(draft):
            raise ValueError("Chat template did not preserve the assistant draft")
        boundary = _boundary_after_text_with_offsets(tokenizer, target_text, draft)
        if boundary is None:
            # When offsets are unavailable, mask the boundary token too if it
            # merges draft text with the following routing suffix.
            prefix_ids = tokenizer(target_text[:end], add_special_tokens=False)["input_ids"]
            common = _common_prefix_len(prefix_ids, target_ids)
            boundary = common + int(common < len(prefix_ids))
        label_boundary += boundary
    labels = [-100] * label_boundary + input_ids[label_boundary:]
    meaningful = bool(str(response_messages[0].get("content") or "").strip()
                      or response_messages[0].get("tool_calls"))
    supervised = sum(y != -100 for y in labels) if meaningful else 0
    stats = {"total_tokens": len(input_ids), "prompt_boundary": len(prompt_ids),
             "label_boundary": label_boundary, "supervised_tokens": supervised,
             "truncated": int(len(input_ids) > max_seq_len)}
    if stats["truncated"] or not supervised or not prompt_ids:
        return None, stats
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}, stats


def build_anchor_features(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_len: int,
    mode: str,
    tools=None,
) -> Tuple[List[Dict[str, List[int]]], Dict[str, float]]:
    """Tokenize anchor rows and summarize masking/truncation diagnostics."""
    features: List[Dict[str, List[int]]] = []
    total_supervised = 0
    total_tokens = 0
    dropped = 0
    truncated = 0
    for row in rows:
        feature, stats = tokenize_anchor_row(
            row=row,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            tools=tools,
        )
        total_tokens += stats["total_tokens"]
        truncated += stats["truncated"]
        if feature is None:
            dropped += 1
        else:
            features.append(feature)
            total_supervised += stats["supervised_tokens"]

    n_kept = len(features)
    return features, {
        "n_rows": float(len(rows)),
        "n_kept": float(n_kept),
        "n_dropped_no_target": float(dropped - truncated),
        "n_dropped_truncated": float(truncated),
        "mean_tokens": total_tokens / max(1, len(rows)),
        "mean_supervised_tokens": total_supervised / max(1, n_kept),
    }


# --- Qwen3.5 tool_call arguments normalization (appended patch) ---
import json as _json_qwen35


def _normalize_messages_qwen35(value):
    if not (isinstance(value, list) and value and isinstance(value[0], dict)):
        return value
    normalized = []
    for message in value:
        new_message = dict(message)
        calls = message.get("tool_calls")
        new_calls = []
        for call in calls or []:
            new_call = dict(call)
            function = dict(new_call.get("function", {}))
            arguments = function.get("arguments")
            function["arguments"] = _json_qwen35.loads(arguments or "{}") if isinstance(arguments, str) else arguments
            new_call["function"] = function
            new_calls.append(new_call)
        new_message["tool_calls"] = new_calls if calls else calls
        normalized.append(new_message)
    return normalized


_render_chat_pre_qwen35 = _render_chat


def _render_chat(*args, **kwargs):
    args = tuple(_normalize_messages_qwen35(a) for a in args)
    kwargs = {k: _normalize_messages_qwen35(v) for k, v in kwargs.items()}
    return _render_chat_pre_qwen35(*args, **kwargs)
