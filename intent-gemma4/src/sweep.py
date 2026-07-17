#!/usr/bin/env python3
"""Benchmark several models against the same fixture and print a comparison.

Each model in the sweep config is loaded in turn (and freed before the next, so
this is safe on memory-constrained hardware), runs the benchmark fixture against
the same generated tool set, and contributes one row to a table of accuracy +
latency + throughput.

IMPORTANT: latency/throughput are only meaningful on the TARGET hardware (a
Pi 5 or mini PC). Run this there. Accuracy is comparable anywhere.

    python3 src/sweep.py --passes 3 --n-threads 4
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ruamel.yaml import YAML

import benchmark
from gemma4_recognizer import Gemma4Recognizer
from hass_api import HomeInfo
from intent_tools import build_tools, load_metadata
from models import Area, Entity, Floor

BASE_DIR = Path(__file__).resolve().parent


def _synthetic_home(domains: List[str]) -> HomeInfo:
    """A small synthetic home in the given domains, to measure a *filtered*
    (realistic) tool set rather than the full 40-tool worst case."""
    entities = {}
    for domain in domains:
        for i in range(3):
            entity_id = f"{domain}.device_{i}"
            entities[entity_id] = Entity(
                entity_id=entity_id, name=f"{domain} {i}", area_id="a1"
            )
    areas = {
        "a1": Area(area_id="a1", name="Living Room", floor_id="f1"),
        "a2": Area(area_id="a2", name="Kitchen", floor_id="f1"),
    }
    floors = {"f1": Floor(floor_id="f1", name="Ground")}
    return HomeInfo(entities=entities, areas=areas, floors=floors)


def _load_configs(path: Path) -> List[Dict[str, Any]]:
    data = YAML(typ="safe").load(path.read_text(encoding="utf-8")) or {}
    return data.get("configs") or []


def _free(recognizer: Optional[Gemma4Recognizer]) -> None:
    if recognizer is None:
        return
    llm = getattr(recognizer, "llm", None)
    if llm is not None and hasattr(llm, "close"):
        try:
            llm.close()
        except Exception:  # noqa: BLE001
            pass
    del recognizer
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=str(BASE_DIR / "sweep.yaml"))
    parser.add_argument("--intents", default=str(BASE_DIR / "intents.yaml"))
    parser.add_argument("--fixture", default=str(BASE_DIR / "benchmark.yaml"))
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument(
        "--n-threads",
        type=int,
        default=0,
        help="Default CPU threads (0 = llama.cpp default); per-config overrides win",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--state-dir",
        default=str(BASE_DIR.parent / "local" / "sweep"),
        help="Where to cache each model's prefix state",
    )
    parser.add_argument("--json", help="Optional path to write full results JSON")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configs = _load_configs(Path(args.configs))
    if not configs:
        print(f"No configs in {args.configs}", file=sys.stderr)
        sys.exit(1)

    metadata = load_metadata(args.intents)
    # Default: full (unfiltered) tool set so accuracy is comparable across
    # models. A config may set `home_domains: [light, switch, ...]` to measure a
    # filtered, realistic-home tool set instead.
    full_tools = build_tools(metadata, None)
    fixture = benchmark.load_fixture(args.fixture)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    full: List[Dict[str, Any]] = []
    for i, config in enumerate(configs, start=1):
        label = config.get("label") or config.get("filename") or f"config-{i}"
        print(f"[{i}/{len(configs)}] {label} ...", flush=True)
        state_path = state_dir / f"{label}.bin"
        recognizer: Optional[Gemma4Recognizer] = None
        try:
            recognizer = Gemma4Recognizer(
                state_path=state_path,
                repo_id=config["repo_id"],
                filename=config["filename"],
                n_threads=config.get("n_threads", args.n_threads) or None,
                max_tokens=config.get("max_tokens", args.max_tokens),
                enable_thinking=config.get("enable_thinking", False),
                chat_format=config.get("chat_format"),
                flash_attn=config.get("flash_attn", False),
                prompt_lookup=config.get("prompt_lookup", 0),
                debug=args.debug,
            )
            home_domains = config.get("home_domains")
            tools = (
                build_tools(metadata, _synthetic_home(home_domains))
                if home_domains
                else full_tools
            )
            # build_prefix=False: the benchmark builds its own warm prefix, so
            # don't pay the prefix eval twice (matters on slow CPUs).
            recognizer.load(tools, build_prefix=False)
            result = benchmark.run(recognizer, fixture, passes=args.passes)
            summary = result["summary"]
            rows.append(
                {
                    "label": label,
                    "filename": config["filename"],
                    "num_tools": summary["num_tools"],
                    "pass_rate": summary["pass_rate"],
                    "num_passed": summary["num_passed"],
                    "num_sentences": summary["num_sentences"],
                    "median_latency_ms": summary["median_latency_ms"],
                    "tokens_per_second": summary["overall_tokens_per_second"],
                    "rebuild_seconds": summary["rebuild_seconds"],
                }
            )
            full.append({"label": label, "config": config, "result": result})
        except Exception as err:  # noqa: BLE001
            print(f"    FAILED: {type(err).__name__}: {err}", file=sys.stderr)
            rows.append({"label": label, "filename": config.get("filename"), "error": str(err)})
        finally:
            _free(recognizer)
            # Prefix-state files are large (~0.5 GB); don't accumulate them.
            for stale in (state_path, state_path.with_suffix(".sha256")):
                try:
                    stale.unlink()
                except OSError:
                    pass

    _print_table(rows)

    if args.json:
        Path(args.json).write_text(json.dumps(full, indent=2), encoding="utf-8")
        print(f"\nWrote full results to {args.json}")


def _print_table(rows: List[Dict[str, Any]]) -> None:
    def fmt(value, spec, dash="-"):
        return format(value, spec) if value is not None else dash

    ok = [r for r in rows if "error" not in r]
    # Best accuracy first, then fastest.
    ok.sort(
        key=lambda r: (-(r["pass_rate"] or 0), r["median_latency_ms"] or 1e9)
    )

    header = f"{'model':28} {'acc':>9} {'median ms':>10} {'tok/s':>7} {'prefix s':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in ok:
        acc = f"{r['num_passed']}/{r['num_sentences']}"
        print(
            f"{r['label'][:28]:28} {acc:>9} "
            f"{fmt(r['median_latency_ms'], '.0f'):>10} "
            f"{fmt(r['tokens_per_second'], '.1f'):>7} "
            f"{fmt(r['rebuild_seconds'], '.1f'):>9}"
        )
    for r in rows:
        if "error" in r:
            print(f"{r['label'][:28]:28} {'ERROR':>9}  {r['error']}")


if __name__ == "__main__":
    main()
