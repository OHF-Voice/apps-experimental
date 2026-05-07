import json
import re
from typing import Any, Dict, List, Tuple

TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:([a-zA-Z0-9_]+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    text = _normalize_gemma_tool_text(text)
    calls = []

    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1)
        raw_args = match.group(2).strip()
        args = {}

        if raw_args:
            for part in _split_args(raw_args):
                key, value = part.split(":", 1)
                args[key.strip()] = _parse_value(value)

        calls.append((name, args))

    return calls


def _normalize_gemma_tool_text(text: str) -> str:
    return text.replace('<|"|>', '"').replace("<|'|>", "'")


def _parse_value(value: str):
    value = _normalize_gemma_tool_text(value.strip())

    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]

    if re.fullmatch(r"-?\d+", value):
        return int(value)

    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    return value


def _split_args(raw_args: str) -> List[str]:
    parts = []
    buf = []
    in_string = False
    escape = False

    for ch in raw_args:
        if escape:
            buf.append(ch)
            escape = False
            continue

        if ch == "\\":
            buf.append(ch)
            escape = True
            continue

        if ch == '"':
            buf.append(ch)
            in_string = not in_string
            continue

        if ch == "," and not in_string:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf.clear()
            continue

        buf.append(ch)

    part = "".join(buf).strip()
    if part:
        parts.append(part)

    return parts
