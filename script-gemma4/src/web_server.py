import json
import logging
import threading
from pathlib import Path
from typing import Optional, Union

from flask import Flask, Response, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

import benchmark
from benchmark import Fixture
from const import AppState

_LOGGER = logging.getLogger(__name__)

MAX_PASSES = 10


def make_web_server(
    state: AppState, benchmark_fixture_path: Union[str, Path]
) -> Flask:
    flask_app = Flask(__name__)
    flask_app.secret_key = "e603c0d4-18d8-4018-ad45-4c89a0aa9941"

    flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app, x_proto=1, x_host=1)  # type: ignore[method-assign]
    flask_app.wsgi_app = IngressPrefixMiddleware(flask_app.wsgi_app)  # type: ignore[method-assign]

    tools_list = [t.tool for t in state.tools.values()]
    tools_json = json.dumps(tools_list, indent=2)

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
        return Response(tools_json, mimetype="application/json")

    @flask_app.route("/health")
    def health():
        return {"status": "ok"}, 200

    @flask_app.route("/benchmark", methods=["GET"])
    def benchmark_page():
        return Response(BENCHMARK_HTML, mimetype="text/html")

    @flask_app.route("/benchmark/run", methods=["POST"])
    def benchmark_run():
        if fixture is None:
            return {"error": fixture_error or "No benchmark fixture"}, 500

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


BENCHMARK_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma 4 Benchmark</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #1c1c1c; }
  h1 { font-size: 1.4rem; }
  .controls { margin: 1rem 0; display: flex; gap: 0.75rem; align-items: center; }
  button { padding: 0.4rem 1rem; font-size: 1rem; cursor: pointer; }
  button[disabled] { opacity: 0.5; cursor: default; }
  input { width: 4rem; padding: 0.3rem; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; font-size: 0.9rem; }
  th, td { border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; vertical-align: top; }
  th { background: #f3f3f3; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .pass { color: #1a7f37; font-weight: 600; }
  .fail { color: #cf222e; font-weight: 600; }
  code { background: #f3f3f3; padding: 0 0.2rem; border-radius: 3px; white-space: pre-wrap; }
  .summary { background: #f7f7f7; border: 1px solid #ddd; border-radius: 6px;
             padding: 0.75rem 1rem; margin-top: 1rem; }
  .summary div { margin: 0.15rem 0; }
  .muted { color: #666; font-size: 0.85rem; }
  #status { margin-left: 0.5rem; }
</style>
</head>
<body>
<h1>Gemma 4 Script Runner &mdash; Benchmark</h1>
<p class="muted">Runs a fixed tool/sentence fixture against the loaded model.
Live state is snapshotted and restored, so this does not disturb the running
assistant &mdash; but voice handling pauses for the benchmark's duration.</p>
<div class="controls">
  <label>Passes <input id="passes" type="number" min="1" max="10" value="3"></label>
  <button id="run">Run benchmark</button>
  <span id="status"></span>
</div>
<div id="summary"></div>
<div id="results"></div>
<script>
const runBtn = document.getElementById("run");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const resultsEl = document.getElementById("results");

function fmt(n, d) { return (n === null || n === undefined) ? "&mdash;" : n.toFixed(d); }
function esc(s) {
  return String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}
function calls(list) {
  if (!list || !list.length) return "<em>(none)</em>";
  return list.map(c => "<code>" + esc(c.name) + " " +
    esc(JSON.stringify(c.args || {})) + "</code>").join("<br>");
}

async function run() {
  runBtn.disabled = true;
  statusEl.textContent = "Running…";
  summaryEl.innerHTML = "";
  resultsEl.innerHTML = "";
  const passes = parseInt(document.getElementById("passes").value, 10) || 3;
  const url = window.location.pathname.replace(/\\/+$/, "") + "/run";
  const started = performance.now();
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({passes})
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || ("HTTP " + resp.status));
    }
    const data = await resp.json();
    render(data);
    const elapsed = ((performance.now() - started) / 1000).toFixed(1);
    statusEl.textContent = "Done in " + elapsed + "s";
  } catch (e) {
    statusEl.textContent = "Error: " + e.message;
  } finally {
    runBtn.disabled = false;
  }
}

function render(data) {
  const s = data.summary;
  const c = s.config || {};
  summaryEl.innerHTML =
    "<div class='summary'>" +
    "<div><b>Correctness:</b> " + s.num_passed + " / " + s.num_sentences +
      " passed (" + fmt(s.pass_rate * 100, 0) + "%)</div>" +
    "<div><b>Latency (median of per-sentence medians):</b> " +
      fmt(s.median_latency_ms, 0) + " ms &nbsp; " +
      "<span class='muted'>mean " + fmt(s.mean_latency_ms, 0) +
      " ms, max " + fmt(s.max_latency_ms, 0) + " ms</span></div>" +
    "<div><b>Throughput:</b> " + fmt(s.overall_tokens_per_second, 1) +
      " tok/s &nbsp; <span class='muted'>prefix rebuild " +
      fmt(s.rebuild_seconds, 2) + " s, " + s.passes + " pass(es), " +
      s.num_tools + " tools</span></div>" +
    "<div class='muted'><b>Config:</b> " + esc(c.filename || "?") +
      " &middot; n_ctx=" + c.n_ctx + " &middot; n_threads=" + c.n_threads +
      " &middot; max_tokens=" + c.max_tokens + " &middot; temp=" + c.temperature +
      " &middot; draft=" + (c.draft_model || "none") + "</div>" +
    "</div>";

  let rows = "";
  for (const r of data.sentences) {
    rows +=
      "<tr>" +
      "<td>" + esc(r.text) + "</td>" +
      "<td class='" + (r.passed ? "pass" : "fail") + "'>" +
        (r.passed ? "PASS" : "FAIL") + "</td>" +
      "<td>" + calls(r.expected) + "</td>" +
      "<td>" + calls(r.got) + "</td>" +
      "<td class='num'>" + fmt(r.median_latency_ms, 0) + "</td>" +
      "<td class='num muted'>" + fmt(r.min_latency_ms, 0) + " / " +
        fmt(r.max_latency_ms, 0) + "</td>" +
      "<td class='num'>" + r.completion_tokens + "</td>" +
      "<td class='num'>" + fmt(r.tokens_per_second, 1) + "</td>" +
      "</tr>";
  }
  resultsEl.innerHTML =
    "<table><thead><tr>" +
    "<th>Sentence</th><th>Result</th><th>Expected</th><th>Got</th>" +
    "<th>Median ms</th><th>min / max</th><th>Tokens</th><th>tok/s</th>" +
    "</tr></thead><tbody>" + rows + "</tbody></table>";
}

runBtn.addEventListener("click", run);
</script>
</body>
</html>
"""
