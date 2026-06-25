"""Latency + correctness benchmark over a fixed sentence fixture.

The fixture (sentences + expected intents) is independent of any live Home
Assistant instance so results are reproducible and comparable across machines
and across model/config changes. Latency is measured against the model's
*actual* loaded tools by default (so it reflects the real, possibly large, tool
prefix); a fixture may override ``tools`` for an isolated measurement.

Each sentence's ``expected`` is the list of Home Assistant intents (name +
slots) that should result after mapping the model's tool calls through
``intent_mapping.tool_to_intent`` -- i.e. the end-to-end output of this service.
The actual model calls happen in ``Gemma4Recognizer.run_sentences`` (which
snapshots and restores live state so benchmarking never disturbs the running
assistant); this module loads the fixture, maps tool calls to intents, and
scores them against the expected ones.
"""

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ruamel.yaml import YAML

from intent_mapping import tool_to_intent

_yaml = YAML(typ="safe")

INTENT = Tuple[str, Dict[str, Any]]


@dataclass
class Fixture:
    sentences: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    language: str = "en"


def load_fixture(path: Union[str, Path]) -> Fixture:
    with open(path, "r", encoding="utf-8") as fixture_file:
        data = _yaml.load(fixture_file) or {}

    return Fixture(
        sentences=data.get("sentences") or [],
        tools=data.get("tools"),
        language=data.get("language") or "en",
    )


def _normalize_intents(intents: List[Dict[str, Any]]) -> List[INTENT]:
    """Order-independent, comparable representation of a list of intents."""
    normalized = [(i["name"], i.get("slots") or {}) for i in intents]
    normalized.sort(key=lambda i: (i[0], json.dumps(i[1], sort_keys=True)))
    return normalized


def _tool_calls_to_intents(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    intents: List[Dict[str, Any]] = []
    for call in tool_calls:
        try:
            name, slots = tool_to_intent(call["name"], call.get("args") or {})
        except ValueError:
            # Unmappable tool call (e.g. unknown action) -> no intent.
            continue
        intents.append({"name": name, "slots": slots})

    return intents


def run(recognizer: Any, fixture: Fixture, passes: int = 3) -> Dict[str, Any]:
    """Run the fixture through the recognizer and score it at the intent level.

    Returns a JSON-serializable dict with a ``summary`` and per-``sentences``
    results (median/min/max latency, tokens/sec, expected vs. parsed intents).
    """
    tools = fixture.tools if fixture.tools is not None else (recognizer.tools or [])
    raw = recognizer.run_sentences(
        tools,
        fixture.sentences,
        passes=passes,
        default_language=fixture.language,
    )

    sentence_results: List[Dict[str, Any]] = []
    median_latencies: List[float] = []
    total_completion_tokens = 0
    num_passed = 0

    for fixture_sentence, result in zip(fixture.sentences, raw["sentences"]):
        latencies = [p["latency"] for p in result["passes"]]
        median_latency = statistics.median(latencies)
        # Greedy decoding is deterministic, so output is identical across passes.
        completion_tokens = result["passes"][-1]["completion_tokens"] or 0
        tps = (completion_tokens / median_latency) if median_latency > 0 else None

        got_intents = _tool_calls_to_intents(result["tool_calls"])
        expected = _normalize_intents(fixture_sentence.get("expected") or [])
        got = _normalize_intents(got_intents)
        passed = expected == got
        if passed:
            num_passed += 1

        median_latencies.append(median_latency)
        total_completion_tokens += completion_tokens

        sentence_results.append(
            {
                "text": result["text"],
                "language": result["language"],
                "passed": passed,
                "expected": [{"name": n, "slots": s} for n, s in expected],
                "got": got_intents,
                "tool_calls": result["tool_calls"],
                "median_latency_ms": median_latency * 1000.0,
                "min_latency_ms": min(latencies) * 1000.0,
                "max_latency_ms": max(latencies) * 1000.0,
                "completion_tokens": completion_tokens,
                "tokens_per_second": tps,
                "content": result["content"],
            }
        )

    num_sentences = len(sentence_results)
    total_latency = sum(median_latencies)
    summary = {
        "num_sentences": num_sentences,
        "num_passed": num_passed,
        "pass_rate": (num_passed / num_sentences) if num_sentences else None,
        "passes": raw["passes"],
        "num_tools": raw["num_tools"],
        "rebuild_seconds": raw["rebuild_seconds"],
        "mean_latency_ms": (
            statistics.fmean(median_latencies) * 1000.0 if median_latencies else None
        ),
        "median_latency_ms": (
            statistics.median(median_latencies) * 1000.0 if median_latencies else None
        ),
        "max_latency_ms": (max(median_latencies) * 1000.0 if median_latencies else None),
        "overall_tokens_per_second": (
            (total_completion_tokens / total_latency) if total_latency > 0 else None
        ),
        "config": raw["config"],
    }

    return {"summary": summary, "sentences": sentence_results}
