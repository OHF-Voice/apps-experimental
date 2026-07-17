#!/usr/bin/env python3
"""Generate (and optionally run) an English coverage suite over every slot
combination in ``intents.yaml``.

Every ``slot_combination`` of every supported intent contributes one test
sentence (its ``example``), so running the suite tells you, per intent and per
slot combination, what the model actually recognizes in English. The output is
a normal benchmark fixture, so it plugs straight into the existing harness:

    python3 src/coverage.py                       # write src/coverage.yaml
    python3 src/coverage.py --run --n-threads 4   # write + run, grouped report
    python3 src/sweep.py --fixture src/coverage.yaml   # across the model sweep

Scoring checks the intent name plus the *value* slots that are fully determined
by the sentence -- durations (hours/minutes/seconds, and the start_* a timer was
set with), brightness/percentage/position/volume_level, and temperature. Those
use ``match: subset`` (the model-filled name/area/etc. are ignored). Sentences
whose only slots are home-dependent targeting (name/area/floor/domain/
device_class) or free wildcards (item/search_query/conversation_command) fall
back to ``match: intent`` -- their gold value can't be known from a bare example.
Mixed delta+start timer edits (e.g. "add 1 hour to the 2 hour timer") also stay
intent-only: the example often doesn't even state the start duration.

Each generated entry's ``expected`` is the Home Assistant intent that should
result *after* tool->intent mapping. That mapping resolves each tool's short
alias (``intent_tools.TOOL_NAMES``) back to its intent, and folds the split
tools back together (open/close -> HassTurnOn/Off, lock/unlock ->
HassTurnOn/Off), so the open/close examples under HassTurnOn/Off correctly
expect HassTurnOn/Off.

Intents in ``intent_tools.INTENT_EXCLUDES`` are never exposed as tools (they are
routed elsewhere -- e.g. the shopping-list intents go through the todo intents),
so the model cannot emit them. Their sentences are still listed (so coverage is
visible) but annotated ``no_tool: true``; expect them to "fail" recognition.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ruamel.yaml import YAML

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from intent_tools import INTENT_EXCLUDES, SLOT_EXCLUDES, load_metadata  # noqa: E402


# Scalar "N%" / "N percent" value slots (only one is ever present per combo).
_PERCENT_SLOTS = ("brightness", "percentage", "position", "volume_level")
_DURATION_UNITS = (("hour", "hours"), ("minute", "minutes"), ("second", "seconds"))


def _first_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None  # type: ignore[return-value]


def extract_value_slots(text: str, slots: List[str]) -> Dict[str, Any]:
    """Slot values that are unambiguously determined by the example sentence.

    Targeting slots (name/area/floor/domain/device_class) and free wildcards
    (item/search_query/conversation_command/volume_step/state/color) are skipped:
    their value is home-dependent, model-dependent, or simply not derivable from
    a one-line example. Returns {} when nothing can be checked (-> intent-only).
    """
    text = text.lower()
    slot_set = set(slots)
    values: Dict[str, Any] = {}

    for slot in _PERCENT_SLOTS:
        if slot in slot_set:
            number = _first_int(r"(\d+)\s*(?:%|percent)", text)
            if number is not None:
                values[slot] = number

    if "temperature" in slot_set:
        number = _first_int(r"(\d+)\s*kelvin", text) or _first_int(
            r"(\d+)\s*degree", text
        )
        if number is not None:
            values["temperature"] = number

    # Durations. A combo is either delta-only (hours/minutes/seconds) or
    # start-only (start_*); the mixed delta+start edit combos are too ambiguous
    # to gold-label from the example (and usually don't state the start), so we
    # leave their durations unchecked.
    delta = slot_set & {"hours", "minutes", "seconds"}
    start = slot_set & {"start_hours", "start_minutes", "start_seconds"}
    target_keys = None
    if delta and not start:
        target_keys = {unit: plural for unit, plural in _DURATION_UNITS}
    elif start and not delta:
        target_keys = {unit: f"start_{plural}" for unit, plural in _DURATION_UNITS}
    if target_keys is not None:
        for unit, slot in target_keys.items():
            number = _first_int(rf"(\d+)\s*{unit}", text)
            if number is not None:
                values[slot] = number

    return values


def _examples(combination: Dict[str, Any]) -> List[str]:
    """An ``example`` may be a single string or a list; normalize to a list."""
    example = combination.get("example")
    if example is None:
        return []
    if isinstance(example, str):
        return [example]
    return [str(e) for e in example]


def build_entries(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One entry per (intent, slot_combination, example), de-duplicated.

    De-dup is on (lowercased text, intent): the same example sentence can appear
    under several combinations of one intent (e.g. repeated timer phrasings).
    A sentence shared by *different* intents is intentionally kept once per
    intent -- those are genuine ambiguities worth seeing in the report.
    """
    entries: List[Dict[str, Any]] = []
    seen: set = set()
    for intent, info in metadata.items():
        if not info.get("supported"):
            continue
        no_tool = intent in INTENT_EXCLUDES
        for comb_name, comb in (info.get("slot_combinations") or {}).items():
            for text in _examples(comb):
                key = (text.strip().lower(), intent)
                if key in seen:
                    continue
                seen.add(key)
                comb_slots = list(comb.get("slots") or [])
                # Slots deliberately omitted from the generated tool
                # (intent_tools.SLOT_EXCLUDES, e.g. the timer start_* slots)
                # can never be filled -- don't gold-check them; the sentence
                # still scores at the intent level.
                scorable_slots = [
                    s for s in comb_slots if (intent, s) not in SLOT_EXCLUDES
                ]
                values = extract_value_slots(text, scorable_slots)
                expected: Dict[str, Any] = {"name": intent}
                if values:
                    expected["slots"] = values
                entry: Dict[str, Any] = {
                    "text": text,
                    "intent": intent,
                    "combination": comb_name,
                    "importance": comb.get("importance", "required"),
                    "slots": comb_slots,
                    "match": "subset" if values else "intent",
                    "expected": [expected],
                }
                if no_tool:
                    entry["no_tool"] = True
                entries.append(entry)
    return entries


def write_fixture(entries: List[Dict[str, Any]], path: Path) -> None:
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 200
    header = (
        "# AUTO-GENERATED by coverage.py from intents.yaml -- do not edit by hand.\n"
        "# One sentence per slot combination of every supported intent. `expected`\n"
        "# is the HA intent after tool->intent mapping. match: subset checks the\n"
        "# intent name plus the value slots derivable from the sentence (durations,\n"
        "# percentages, temperature), ignoring model-filled name/area/etc.; match:\n"
        "# intent checks the name only (home-dependent / wildcard slots). Entries\n"
        "# with `no_tool: true` are not exposed as tools (intent_tools.\n"
        "# INTENT_EXCLUDES) and are expected to go unrecognized. Annotation keys\n"
        "# (intent/combination/importance/slots/no_tool) are ignored by the harness.\n"
        "#\n"
        "# Regenerate:  python3 src/coverage.py\n\n"
    )
    with open(path, "w", encoding="utf-8") as out:
        out.write(header)
        yaml.dump({"language": "en", "sentences": entries}, out)


def _summarize(entries: List[Dict[str, Any]]) -> None:
    from collections import Counter

    by_intent = Counter(e["intent"] for e in entries)
    no_tool = sorted({e["intent"] for e in entries if e.get("no_tool")})
    print(
        f"{len(entries)} sentences across {len(by_intent)} intents "
        f"({sum(by_intent.values())} combinations)."
    )
    if no_tool:
        print(f"Not exposed as tools (expected unrecognized): {no_tool}")


def _run(
    fixture_path: Path, n_threads: int, passes: int, json_path: Optional[Path] = None
) -> None:
    """Load the model against the FULL tool set and report coverage per intent.

    Full (unfiltered) tools so coverage reflects the model, not a particular
    home's exposed entities (those gate tools out; see the live benchmark).
    """
    import benchmark
    from gemma4_recognizer import Gemma4Recognizer
    from intent_tools import build_tools

    metadata = load_metadata(BASE_DIR / "intents.yaml")
    tools = build_tools(metadata, None)
    fixture = benchmark.load_fixture(fixture_path)

    rec = Gemma4Recognizer(
        state_path=str(BASE_DIR.parent / "local" / "coverage_state.bin"),
        n_threads=n_threads or None,
    )
    rec.load(tools, build_prefix=False)
    result = benchmark.run(rec, fixture, passes=passes)

    if json_path is not None:
        import json

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, indent=2)
        print(f"Wrote results to {json_path}")

    # Group per-sentence results by intent (annotations live on fixture.sentences).
    from collections import defaultdict

    groups: Dict[str, List[tuple]] = defaultdict(list)
    for spec, res in zip(fixture.sentences, result["sentences"]):
        groups[spec["intent"]].append((spec, res))

    s = result["summary"]
    print(
        f"\n=== COVERAGE: {s['num_passed']}/{s['num_sentences']} "
        f"({s['pass_rate']:.0%}) over {s['num_tools']} tools ===\n"
    )
    for intent in sorted(groups):
        rows = groups[intent]
        passed = sum(1 for _, r in rows if r["passed"])
        no_tool = rows[0][0].get("no_tool")
        flag = "  [no tool]" if no_tool else ""
        print(f"{intent:28} {passed}/{len(rows)}{flag}")
        for spec, res in rows:
            if res["passed"]:
                continue
            got = [c["name"] for c in res["tool_calls"]] or ["(none)"]
            detail = f" -> {got}"
            if spec["match"] == "subset":
                want = spec["expected"][0].get("slots") or {}
                gots = res["got"][0]["slots"] if res["got"] else {}
                detail = f" want {want} got {got}{gots or ''}"
            print(f"    FAIL [{spec['combination']}] {spec['text']!r}{detail}")
    print("\nDONE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", default=str(BASE_DIR / "intents.yaml"))
    parser.add_argument("--out", default=str(BASE_DIR / "coverage.yaml"))
    parser.add_argument(
        "--run",
        action="store_true",
        help="After writing, load the model and print a per-intent coverage report",
    )
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument(
        "--json",
        help="With --run, also write the full benchmark results to this JSON file "
        "(so accuracy can be diffed across tool-set changes)",
    )
    args = parser.parse_args()

    metadata = load_metadata(args.intents)
    entries = build_entries(metadata)
    out_path = Path(args.out)
    write_fixture(entries, out_path)
    print(f"Wrote {out_path}")
    _summarize(entries)

    if args.run:
        _run(
            out_path,
            args.n_threads,
            args.passes,
            json_path=Path(args.json) if args.json else None,
        )


if __name__ == "__main__":
    main()
