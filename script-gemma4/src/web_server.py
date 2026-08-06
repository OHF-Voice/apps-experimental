import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from flask import Flask, Response, jsonify, render_template, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

import benchmark
from benchmark import Fixture
from const import AppState
from hass_api import Tool
from tool_mapping import map_tool_call, required_fields

_LOGGER = logging.getLogger(__name__)

MAX_PASSES = 10

# Names shown per enum field before collapsing into "+N more".
ENUM_SAMPLE = 5


def _describe_fields(tool: Tool) -> List[Dict[str, Any]]:
    """Summarize a tool's fields for the web UI."""
    function = tool.tool.get("function") or {}
    params = function.get("parameters") or {}
    props = params.get("properties") or {}
    required = required_fields(tool)

    fields: List[Dict[str, Any]] = []
    for field_name, schema in sorted(props.items()):
        schema = schema or {}
        enum = schema.get("enum") or []
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
                "mapped": bool(tool.name_map.get(field_name)),
            }
        )

    return fields


def _describe_script(tool: Tool) -> Dict[str, Any]:
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
        "fields": _describe_fields(tool),
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
        scripts = [_describe_script(tool) for tool in state.all_tools]
        return render_template(
            "home.html",
            targeted=[s for s in scripts if s["exposed"]],
            untargeted=[s for s in scripts if not s["exposed"]],
        )

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
                        "missing_required": [],
                    }
                )
                continue

            call = map_tool_call(tool, tool_args)
            calls.append(
                {
                    "script_id": call.script_id,
                    "unknown_tool": False,
                    "can_run": call.can_run,
                    "variables": call.variables,
                    "unresolved": call.unresolved,
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
