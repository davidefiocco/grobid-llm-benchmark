"""Lenient JSON parsing for model responses.

Models may wrap the object in prose or markdown fences, or hit a token cap mid-object.
These helpers recover the largest valid object so a partial reference list is kept rather
than discarding the whole article.
"""

from __future__ import annotations

import json
import re


def _repair_truncated_json(text: str) -> dict | None:
    """Recover a JSON object cut off mid-generation by closing open strings/brackets.

    Prefers cutting after the last completed array element or object member so only the
    partial trailing item is dropped.
    """
    s = text.strip()
    start = s.find("{")
    if start == -1:
        return None
    s = s[start:]

    in_str = False
    escaped = False
    depth: list[str] = []
    last_safe = 0
    for i, ch in enumerate(s):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
                last_safe = i + 1
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if depth:
                depth.pop()
            last_safe = i + 1
        elif ch == ",":
            last_safe = i + 1
        elif ch.isdigit() or ch in "aeflnrstu.-+":
            last_safe = i + 1

    def _try(candidate: str) -> dict | None:
        cand = candidate.rstrip()
        while cand and cand[-1] in ",:":
            cand = cand[:-1].rstrip()
        d: list[str] = []
        instr = False
        esc = False
        for c in cand:
            if instr:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    instr = False
                continue
            if c == '"':
                instr = True
            elif c in "{[":
                d.append("}" if c == "{" else "]")
            elif c in "}]":
                if d:
                    d.pop()
        if instr:
            cand += '"'
        cand += "".join(reversed(d))
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            return None

    for candidate in (s[:last_safe], s):
        out = _try(candidate)
        if out is not None:
            return out
    return None


def coerce_json(text: str) -> dict:
    """Parse a JSON object out of a model response, tolerating fences/prose/truncation."""
    obj, _ = coerce_json_ex(text)
    return obj


def coerce_json_ex(text: str) -> tuple[dict, bool]:
    """Like :func:`coerce_json`, but also return whether the object had to be *repaired*
    from a truncated response (a signal that the reference list may be cut short)."""
    text = text.strip()
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1)), False
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0)), False
        except json.JSONDecodeError:
            pass
    repaired = _repair_truncated_json(text)
    if repaired is not None:
        return repaired, True
    raise ValueError("No JSON object found in model response")
