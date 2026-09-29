"""Explicit final-answer extraction without incidental derivation-number fallback.

Mathematical correctness and typographical conventions are separate: surrounding
math delimiters, Markdown emphasis, whitespace and sentence punctuation do not
change the submitted answer. Conflicting declarations and prose after it do.
"""
from __future__ import annotations

import re
from fractions import Fraction
from functools import lru_cache


def unbox(text: str) -> str | None:
    text = text.strip()
    opening = re.match(r"^\\boxed\s*\{", text)
    if not opening:
        return None
    depth = 1
    for i in range(opening.end(), len(text)):
        escaped = (len(text[:i]) - len(text[:i].rstrip("\\"))) % 2
        if text[i] == "{" and not escaped:
            depth += 1
        elif text[i] == "}" and not escaped:
            depth -= 1
            if depth == 0:
                return text[opening.end():i].strip() if not text[i + 1:].strip() else None
    return None


def _unwrap_presentation(text: str) -> str:
    """Remove only balanced, whole-value presentation wrappers."""
    wrappers = (("$$", "$$"), ("$", "$"), (r"\[", r"\]"),
                (r"\(", r"\)"), ("**", "**"), ("__", "__"), ("`", "`"))
    text = text.strip()
    while text:
        if text.endswith((".", "。")):
            text = text[:-1].rstrip()
            continue
        for left, right in wrappers:
            if len(text) > len(left) + len(right) and text.startswith(left) and text.endswith(right):
                text = text[len(left):-len(right)].strip()
                break
        else:
            return text
    return text


def extract_final(text: str) -> str | None:
    """Require one explicit terminal declaration; tolerate harmless formatting.

    No free-form number or earlier boxed equation is selected as a fallback.
    Generation truncation is checked separately by the rollout validator.
    """
    text = text or ""
    if len(re.findall(r"FINAL_ANSWER\s*:", text, flags=re.I)) != 1:
        return None
    declaration = re.search(r"(?im)^[ \t]*(?:\*\*|__)?FINAL_ANSWER\s*:[ \t]*", text)
    if declaration is None:
        return None
    payload = text[declaration.end():].strip()
    # An emphasis wrapper may enclose the entire declaration instead of just
    # its label/value. Only remove its matching terminal close.
    start = text[declaration.start():declaration.end()].lstrip()
    if start.startswith(("**", "__")):
        marker = start[:2]
        if payload.startswith(marker):
            payload = payload[2:]
        elif payload.endswith(marker):
            payload = payload[:-2]
    answer = unbox(_unwrap_presentation(payload))
    return answer if answer and len(answer) <= 1024 else None


def as_draft(text: str) -> str:
    """Keep the full derivation, removing only its terminal declaration."""
    return "\n".join(line for line in text.splitlines()
                     if not line.strip().startswith("FINAL_ANSWER:")).strip()


def _number(text: str) -> Fraction | None:
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", text):
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError):
            return None
    return None


@lru_cache(maxsize=8192)
def _parse(text: str):
    from math_verify import LatexExtractionConfig, parse
    return parse(r"\boxed{" + text + "}", extraction_config=[LatexExtractionConfig()],
                 fallback_mode="no_fallback", extraction_mode="first_match")


def valid_gold(gold: str) -> bool:
    if not gold or gold.strip().lower() in {"proof", "unknown", "none", "n/a"}:
        return False
    return _number(gold.strip()) is not None or bool(_parse(gold.strip()))


def equivalent(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    p, g = prediction.strip(), str(gold).strip()
    if not p or len(p) > 1024:
        return False
    pn, gn = _number(p), _number(g)
    if pn is not None and gn is not None:
        return pn == gn
    # Missing dependencies are setup errors, not silently wrong answers.
    from math_verify import verify
    gold_parsed, pred_parsed = _parse(g), _parse(p)
    # Do not let decimal rounding accept a near-miss integer/fraction wrapped
    # in LaTeX (the plain-number path above is exact as well).
    from sympy import Float, Rational
    if len(gold_parsed) == len(pred_parsed) == 1 and all(isinstance(x, (Rational, Float)) for x in (gold_parsed[0], pred_parsed[0])):
        return Rational(str(gold_parsed[0])) == Rational(str(pred_parsed[0]))
    return bool(verify(gold_parsed, pred_parsed))


def correct(text: str, gold: str) -> bool:
    return equivalent(extract_final(text), gold)
