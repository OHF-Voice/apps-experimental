import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from flask import Flask, Response, jsonify, render_template, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

import benchmark
import overrides
from benchmark import Fixture
from const import AppState
from hass_api import SatelliteInfo, Tool
from overrides import AREA, ENTITY, FLOOR, NAME_KINDS, NameOverrides, ScriptOverrides
from tool_mapping import map_tool_call, required_fields

_LOGGER = logging.getLogger(__name__)

MAX_PASSES = 10

# Names shown per enum field before collapsing into "+N more".
ENUM_SAMPLE = 5

# Seconds to wait for Home Assistant when re-reading it.
HASS_TIMEOUT = 60.0


def _describe_fields(tool: Tool, names: NameOverrides) -> List[Dict[str, Any]]:
    """Summarize a tool's fields for the web UI."""
    function = tool.tool.get("function") or {}
    params = function.get("parameters") or {}
    props = params.get("properties") or {}
    required = required_fields(tool)

    fields: List[Dict[str, Any]] = []
    for field_name, schema in sorted(props.items()):
        schema = schema or {}
        enum = schema.get("enum") or []
        name_map = tool.name_map.get(field_name) or {}
        # Names that refer to more than one thing. The satellite's area can often
        # narrow one down at recognition time, but not always, so they are worth
        # showing: giving them distinct names on the Names page always works.
        shared = sorted(name for name, ids in name_map.items() if len(ids) > 1)
        fields.append(
            {
                "name": field_name,
                "type": schema.get("type", "string"),
                "format": schema.get("format"),
                "required": field_name in required,
                "enum_count": len(enum),
                "enum_sample": ", ".join(str(value) for value in enum[:ENUM_SAMPLE]),
                "enum_more": max(0, len(enum) - ENUM_SAMPLE),
                # True when the field's names resolve to Home Assistant ids.
                "mapped": bool(name_map),
                "shared_names": shared,
                "shared_sample": ", ".join(shared[:ENUM_SAMPLE]),
                # Set when this field takes names, so the UI can offer the
                # per-field editor and show how many are overridden there.
                "kind": tool.field_kinds.get(field_name),
                "scoped_count": len(names.field_names(tool.name, field_name)),
            }
        )

    return fields


def _describe_script(
    tool: Tool, targeted: bool, names: NameOverrides
) -> Dict[str, Any]:
    function = tool.tool.get("function") or {}
    return {
        "name": tool.name,
        "friendly_name": tool.friendly_name,
        "description": tool.description,
        # What the model is actually told about this script. Falls back to the
        # script's name when it has no description (and no blueprint to borrow
        # one from), which gives the model very little to match on.
        "model_description": function.get("description") or "",
        "has_description": bool(tool.description),
        "exposed": tool.exposed,
        "targeted": targeted,
        # Targeted despite not being exposed to voice in Home Assistant: worth
        # calling out, since this app runs scripts.
        "force_enabled": targeted and not tool.exposed,
        "fields": _describe_fields(tool, names),
    }


def make_web_server(state: AppState, benchmark_fixture_path: Union[str, Path]) -> Flask:
    flask_app = Flask(__name__)
    flask_app.secret_key = "e603c0d4-18d8-4018-ad45-4c89a0aa9941"

    flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app, x_proto=1, x_host=1)  # type: ignore[method-assign]
    flask_app.wsgi_app = IngressPrefixMiddleware(flask_app.wsgi_app)  # type: ignore[method-assign]

    fixture: Optional[Fixture] = None
    fixture_error: Optional[str] = None
    try:
        fixture = benchmark.load_fixture(benchmark_fixture_path)
    except (OSError, ValueError) as err:
        fixture_error = f"Could not load benchmark fixture: {err}"
        _LOGGER.warning(fixture_error)

    @flask_app.context_processor
    def inject_url_for():
        return dict(url_for=url_for)  # pylint: disable=use-dict-literal

    @flask_app.route("/", methods=["GET"])
    def index():
        """Show every script, split by whether the model can call it."""
        scripts = [
            _describe_script(
                tool, state.overrides.scripts.targets(tool), state.overrides.names
            )
            for tool in state.all_tools
        ]
        areas = sorted(
            ({"id": a.area_id, "name": a.name} for a in state.hass_info.areas.values()),
            key=lambda a: a["name"],
        )
        return render_template(
            "home.html",
            targeted=[s for s in scripts if s["targeted"]],
            untargeted=[s for s in scripts if not s["targeted"]],
            can_save=state.overrides_path is not None,
            areas=areas,
        )

    def _refresh_from_hass() -> None:
        """Re-read Home Assistant and rebuild every tool, in place.

        Runs the fetch on the app's event loop, since the web server is a
        separate thread. Name overrides are applied while the tools are built.
        """
        assert state.loop is not None, "No event loop"

        async def fetch():
            hass_info = await state.hass.get_home_info()
            all_tools = await state.hass.get_script_tools(
                hass_info, state.overrides.names
            )
            return hass_info, all_tools

        hass_info, all_tools = asyncio.run_coroutine_threadsafe(
            fetch(), state.loop
        ).result(timeout=HASS_TIMEOUT)

        if not all_tools:
            # Nothing came back, so this is not a trustworthy snapshot: keep
            # what we have rather than pruning against it.
            raise RuntimeError("Home Assistant returned no scripts")

        state.set_home_info(hass_info)
        state.all_tools = all_tools
        if overrides.prune_all(state.overrides, all_tools, hass_info):
            if state.overrides_path is not None:
                overrides.save(state.overrides_path, state.overrides)

    def _apply_targeted() -> List[Tool]:
        """Swap the model onto the currently targeted tools."""
        targeted = state.overrides.scripts.select(state.all_tools)
        future = state.llama_executor.submit(
            state.recognizer.reload, [tool.tool for tool in targeted]
        )
        future.result()
        state.set_targeted(targeted)
        return targeted

    @flask_app.route("/reload", methods=["POST"])
    def reload_from_hass():
        """Re-read Home Assistant, then rebuild the tools and model prefix.

        The app otherwise reads Home Assistant once, at start, so this is how a
        newly exposed script or a renamed entity is picked up.
        """
        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        if not state.reload_lock.acquire(blocking=False):
            return jsonify({"error": "Already applying a change"}), 409

        try:
            _refresh_from_hass()
            targeted = _apply_targeted()
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Failed to reload from Home Assistant")
            return jsonify({"error": f"Could not reload: {err}"}), 500
        finally:
            state.reload_lock.release()

        return jsonify(
            {
                "num_scripts": len(state.all_tools),
                "num_targeted": len(targeted),
            }
        )

    def _requested_names(body: Dict[str, Any]) -> Optional[Set[str]]:
        """The script names the page wants targeted, or None if malformed."""
        names = body.get("targeted")
        if not isinstance(names, list):
            return None

        known = {tool.name for tool in state.all_tools}
        return {str(name) for name in names} & known

    @flask_app.route("/overrides/estimate", methods=["POST"])
    def overrides_estimate():
        """Report what a prospective tool set would cost, before applying it."""
        body = request.get_json(silent=True) or {}
        names = _requested_names(body)
        if names is None:
            return jsonify({"error": "No script list given"}), 400

        tools = [tool.tool for tool in state.all_tools if tool.name in names]
        return jsonify(
            {
                "num_tools": len(tools),
                "n_ctx": state.recognizer.required_n_ctx(tools),
                "current_n_ctx": (
                    state.recognizer.llm.n_ctx()
                    if state.recognizer.llm is not None
                    else None
                ),
                "fixed_n_ctx": state.recognizer.n_ctx is not None,
            }
        )

    @flask_app.route("/overrides", methods=["POST"])
    def overrides_apply():
        """Change which scripts are targeted, then rebuild the model prefix.

        Recognition is unavailable while the prefix is rebuilt, which can take
        minutes on slow hardware, so this is deliberately an explicit action
        rather than something that happens per checkbox.
        """
        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        body = request.get_json(silent=True) or {}
        names = _requested_names(body)
        if names is None:
            return jsonify({"error": "No script list given"}), 400

        # A second apply while one is in flight would race on the model.
        if not state.reload_lock.acquire(blocking=False):
            return jsonify({"error": "Already applying a change"}), 409

        try:
            new_overrides = ScriptOverrides(
                enabled={
                    tool.name
                    for tool in state.all_tools
                    if (tool.name in names) and not tool.exposed
                },
                disabled={
                    tool.name
                    for tool in state.all_tools
                    if (tool.name not in names) and tool.exposed
                },
            )
            previous = state.overrides.scripts
            state.overrides.scripts = new_overrides
            try:
                # Swap on the model worker: it owns the model, and a queued
                # recognition will wait rather than race the rebuild.
                targeted = _apply_targeted()
            except Exception as err:  # pylint: disable=broad-except
                state.overrides.scripts = previous
                _LOGGER.exception("Failed to apply script overrides")
                return jsonify({"error": f"Could not apply: {err}"}), 500

            # Only record the change once the model actually accepted it.
            if state.overrides_path is not None:
                overrides.save(state.overrides_path, state.overrides)

            return jsonify(
                {
                    "num_targeted": len(targeted),
                    "enabled": sorted(new_overrides.enabled),
                    "disabled": sorted(new_overrides.disabled),
                    "n_ctx": (
                        state.recognizer.llm.n_ctx()
                        if state.recognizer.llm is not None
                        else None
                    ),
                }
            )
        finally:
            state.reload_lock.release()

    @flask_app.route("/tools.json", methods=["GET"])
    def tools_json():
        """The tools as given to the model (OpenAI function spec), for debugging."""
        tools = [tool.tool for tool in state.tools.values()]
        return Response(json.dumps(tools, indent=2), mimetype="application/json")

    @flask_app.route("/test", methods=["POST"])
    def test():
        """Recognize one sentence and report what it would run, without running it."""
        body = request.get_json(silent=True) or {}
        text = str(body.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text given"}), 400
        language = str(body.get("language") or "").strip() or "en"

        # Simulates the satellite that heard the command, which is what narrows a
        # name shared by several things.
        area_id = str(body.get("area_id") or "").strip() or None
        satellite = (
            SatelliteInfo(
                area_id=area_id,
                floor_id=state.geometry.area_floors.get(area_id) if area_id else None,
            )
            if area_id
            else None
        )

        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        # Serialize on the model worker, like live recognition and the benchmark.
        start_time = time.monotonic()
        future = state.llama_executor.submit(
            state.recognizer.get_tool_calls, text, language
        )
        tool_calls, response_text = future.result()
        latency_ms = (time.monotonic() - start_time) * 1000.0

        calls = []
        for tool_id, tool_args in tool_calls:
            tool = state.tools.get(tool_id)
            if tool is None:
                # The model can emit a name that is not a tool at all.
                calls.append(
                    {
                        "script_id": tool_id,
                        "unknown_tool": True,
                        "can_run": False,
                        "variables": {},
                        "unresolved": {},
                        "ambiguous": {},
                        "missing_required": [],
                    }
                )
                continue

            call = map_tool_call(tool, tool_args, satellite, state.geometry)
            calls.append(
                {
                    "script_id": call.script_id,
                    "unknown_tool": False,
                    "can_run": call.can_run,
                    "variables": call.variables,
                    "unresolved": call.unresolved,
                    "ambiguous": call.ambiguous,
                    "missing_required": call.missing_required,
                }
            )

        return jsonify(
            {
                "text": response_text,
                "latency_ms": latency_ms,
                "calls": calls,
            }
        )

    def _name_rows() -> Dict[str, List[Dict[str, Any]]]:
        """Every nameable target, with the names the model is currently offered.

        Only entities exposed to Assist reach the enums, so those are the only
        ones worth listing; all areas and floors are always available.
        """
        info = state.hass_info
        names = state.overrides.names
        rows: Dict[str, List[Dict[str, Any]]] = {}

        def row(kind: str, target_id: str, default_names: List[str], extra: str = ""):
            current = names.names_for(kind, target_id, default_names)
            return {
                "id": target_id,
                "default_names": default_names,
                "names": current,
                "overridden": names.is_overridden(kind, target_id),
                "excluded": names.is_overridden(kind, target_id) and not current,
                "extra": extra,
            }

        rows[ENTITY] = [
            row(ENTITY, entity.entity_id, [n for n in entity.names if n], entity.domain)
            for entity in sorted(info.entities.values(), key=lambda e: e.entity_id)
        ]
        rows[AREA] = [
            row(AREA, area.area_id, [n for n in area.names if n])
            for area in sorted(info.areas.values(), key=lambda a: a.area_id)
        ]
        rows[FLOOR] = [
            row(FLOOR, floor.floor_id, [n for n in floor.names if n])
            for floor in sorted(info.floors.values(), key=lambda f: f.floor_id)
        ]
        return rows

    @flask_app.route("/names", methods=["GET"])
    def names_page():
        """Edit what the model may call each entity, area, and floor."""
        return render_template(
            "names.html",
            rows=_name_rows(),
            can_save=state.overrides_path is not None,
        )

    def _default_names(kind: str, target_id: str) -> List[str]:
        """Home Assistant's own names for one target."""
        info = state.hass_info
        if kind == ENTITY:
            target: Any = info.entities.get(target_id)
        elif kind == AREA:
            target = info.areas.get(target_id)
        else:
            target = info.floors.get(target_id)

        return [name for name in target.names if name] if target else []

    @flask_app.route("/names/field", methods=["GET"])
    def names_field_page():
        """Edit names for one script's field only.

        Reached from that field on the Scripts page, so the list is already the
        one that misbehaved: the field's own candidates, after its domain filter.
        """
        script_name = str(request.args.get("script") or "")
        field_key = str(request.args.get("field") or "")
        tool = next((t for t in state.all_tools if t.name == script_name), None)
        if (tool is None) or (field_key not in tool.field_kinds):
            return (
                render_template(
                    "names_field.html", tool=None, script=script_name, field=field_key
                ),
                404,
            )

        kind = tool.field_kinds[field_key]
        scoped = state.overrides.names.field_names(script_name, field_key)
        rows = []
        for target_id in tool.field_targets.get(field_key, []):
            default_names = _default_names(kind, target_id)
            # What this target is called everywhere else, which is what an empty
            # per-field override falls back to.
            global_names = state.overrides.names.names_for(
                kind, target_id, default_names
            )
            current = scoped.get(target_id, global_names)
            rows.append(
                {
                    "id": target_id,
                    "global_names": global_names,
                    "names": current,
                    "overridden": target_id in scoped,
                    "excluded": not current,
                }
            )

        return render_template(
            "names_field.html",
            tool=tool,
            script=script_name,
            field=field_key,
            kind=kind,
            rows=sorted(rows, key=lambda r: r["id"]),
            can_save=state.overrides_path is not None,
        )

    @flask_app.route("/names/field", methods=["POST"])
    def names_field_apply():
        """Replace one script field's name overrides, then rebuild."""
        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        body = request.get_json(silent=True) or {}
        script_name = str(body.get("script") or "")
        field_key = str(body.get("field") or "")
        changes = body.get("names")
        if not isinstance(changes, dict):
            return jsonify({"error": "No names given"}), 400

        tool = next((t for t in state.all_tools if t.name == script_name), None)
        if (tool is None) or (field_key not in tool.field_kinds):
            return jsonify({"error": "Unknown script field"}), 404

        # Only ids this field could actually refer to; anything else could never
        # be resolved and would just rot in the file.
        candidates = set(tool.field_targets.get(field_key, []))
        scoped = {
            str(target_id): [
                str(name).strip() for name in target_names if str(name).strip()
            ]
            for target_id, target_names in changes.items()
            if (target_id in candidates) and isinstance(target_names, list)
        }

        if not state.reload_lock.acquire(blocking=False):
            return jsonify({"error": "Already applying a change"}), 409

        previous = dict(state.overrides.names.field_names(script_name, field_key))
        try:
            state.overrides.names.set_field_names(script_name, field_key, scoped)
            _refresh_from_hass()
            _apply_targeted()
        except Exception as err:  # pylint: disable=broad-except
            state.overrides.names.set_field_names(script_name, field_key, previous)
            _LOGGER.exception("Failed to apply names for %s.%s", script_name, field_key)
            return jsonify({"error": f"Could not apply: {err}"}), 500
        finally:
            state.reload_lock.release()

        if state.overrides_path is not None:
            overrides.save(state.overrides_path, state.overrides)

        return jsonify({"num_overridden": len(scoped)})

    @flask_app.route("/names", methods=["POST"])
    def names_apply():
        """Replace name overrides, then rebuild the tools and model prefix.

        Names feed the enums, which are built while reading Home Assistant, so
        this re-reads it rather than trying to patch the tools in place.
        """
        if not state.recognizer.ready:
            return jsonify({"error": "Model is still loading"}), 503

        body = request.get_json(silent=True) or {}
        changes = body.get("names")
        if not isinstance(changes, dict):
            return jsonify({"error": "No names given"}), 400

        info = state.hass_info
        known: Dict[str, Set[str]] = {
            ENTITY: set(info.entities),
            AREA: set(info.areas),
            FLOOR: set(info.floors),
        }

        new_names = NameOverrides()
        for kind in NAME_KINDS:
            for target_id, target_names in (changes.get(kind) or {}).items():
                if target_id not in known[kind]:
                    # Ignore anything that is not a real target: a name has to
                    # resolve to a Home Assistant id to be usable.
                    _LOGGER.debug("Ignoring unknown %s: %s", kind, target_id)
                    continue

                if not isinstance(target_names, list):
                    return (
                        jsonify({"error": f"Names for {target_id} must be a list"}),
                        400,
                    )

                new_names.set_names(
                    kind,
                    str(target_id),
                    [str(name).strip() for name in target_names if str(name).strip()],
                )

        if not state.reload_lock.acquire(blocking=False):
            return jsonify({"error": "Already applying a change"}), 409

        previous = state.overrides.names
        try:
            state.overrides.names = new_names
            _refresh_from_hass()
            targeted = _apply_targeted()
        except Exception as err:  # pylint: disable=broad-except
            state.overrides.names = previous
            _LOGGER.exception("Failed to apply name overrides")
            return jsonify({"error": f"Could not apply: {err}"}), 500
        finally:
            state.reload_lock.release()

        if state.overrides_path is not None:
            overrides.save(state.overrides_path, state.overrides)

        return jsonify(
            {
                "num_overridden": sum(len(v) for v in new_names.by_kind.values()),
                "num_targeted": len(targeted),
            }
        )

    @flask_app.route("/health")
    def health():
        # Deliberately "ok" while the model is still loading: the first boot can
        # spend many minutes downloading it, and the container should not be
        # restarted for being slow. Use /status to see readiness.
        return {"status": "ok"}, 200

    @flask_app.route("/status")
    def status():
        return {"model_loaded": state.recognizer.ready}, 200

    @flask_app.route("/benchmark", methods=["GET"])
    def benchmark_page():
        return render_template("benchmark.html", max_passes=MAX_PASSES)

    @flask_app.route("/benchmark/run", methods=["POST"])
    def benchmark_run():
        if fixture is None:
            return {"error": fixture_error or "No benchmark fixture"}, 500
        if not state.recognizer.ready:
            return {"error": "Model is still loading"}, 503

        body = request.get_json(silent=True) or {}
        try:
            passes = int(body.get("passes", 3))
        except (TypeError, ValueError):
            passes = 3
        passes = max(1, min(MAX_PASSES, passes))

        _LOGGER.info("Running benchmark (passes=%s)", passes)
        # Serialize on the single model worker so a benchmark can't race a live
        # recognition request (the model is not thread-safe). Blocks live voice
        # handling for the benchmark's duration.
        future = state.llama_executor.submit(
            benchmark.run, state.recognizer, fixture, passes
        )
        result = future.result()
        return Response(json.dumps(result), mimetype="application/json")

    return flask_app


def run_web_server(flask_app: Flask, host: str, port: int) -> threading.Thread:
    def run_flask():
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        flask_app.run(host=host, port=port, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)

    return flask_thread


class IngressPrefixMiddleware:
    """Ingress fix for Home Assistant app web UI."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        ingress_path = environ.get("HTTP_X_INGRESS_PATH", "")
        if ingress_path:
            environ["SCRIPT_NAME"] = ingress_path
            path_info = environ.get("PATH_INFO", "")
            if path_info.startswith(ingress_path):
                environ["PATH_INFO"] = path_info[len(ingress_path) :] or "/"
        return self.app(environ, start_response)
