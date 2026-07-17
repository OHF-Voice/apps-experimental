# Gemma 4 Intent Recognizer

Recognizes [Home Assistant intents][intents] locally with [Gemma 4][gemma4]
running on [llama.cpp][] (CPU only), and emits them as [Wyoming][] `Intent`
events. Use it as the intent recognizer for a [voice assistant
pipeline][voice] when you want fully local recognition that is more flexible
than templated sentences.

## How it works

Intent recognition is recast as **LLM tool calling**:

1. Each supported Home Assistant intent becomes one small OpenAI-style
   function tool (e.g. `turn_on`, `set_temperature`, `set_timer`). Broad
   intents are split into language-shaped tools: "open the blinds" and "lock
   the door" call different tools that both map back to `HassTurnOn`.
2. The tool set is generated from the intent metadata and **filtered by your
   home**: an intent whose entity domains have no exposed entities is dropped
   entirely, and the `name`/`area`/`floor`/`device_class` parameters are
   constrained to the actual exposed entity names, areas, and floors (with
   aliases). The model can only produce names that exist.
3. The model's tool call is mapped back to a Home Assistant intent + slots and
   emitted as a Wyoming `Intent` event.

Only entities **exposed to Assist** are included. Expose entities under
*Settings → Voice assistants → Expose*.

### Latency

The fixed prompt (system prompt + tools) is evaluated once and the resulting
llama.cpp state is **persisted to disk**, so restarts skip the expensive
prefix build and each utterance only evaluates the sentence itself. Recent
recognitions are also cached, so repeating a command is instant.

## First start

On first start the model (several GB) is downloaded from Hugging Face and the
tools prefix is built. This can take a long time — tens of minutes on a
Raspberry Pi 5. The web UI is available immediately and shows a "model
loading" notice until recognition is ready. Subsequent starts restore the
persisted prefix in seconds.

Set the `hf_token` option to a Hugging Face API token for faster, more
reliable model downloads.

## Web UI

Open the add-on's web UI (ingress):

- **Home**: a test box — type a sentence and see the recognized intent, its
  description, and the extracted slot values — plus tables of the exposed
  entity, area, and floor names the tools were built from. *Refresh from HA*
  re-reads Home Assistant and rebuilds the tools (do this after exposing new
  entities or renaming things).
- **Tools**: the generated tools as YAML. Saving writes an override file that
  replaces auto-generation (also on restart). Overrides are validated: tool
  names must map to Home Assistant intents. Delete `/data/tools.yaml` to
  return to auto-generation.
- **Benchmark**: runs a fixed sentence set against the loaded model and
  reports per-sentence correctness, latency, and tokens/second. Recognition
  pauses while it runs.

## Supported intents

All supported intents and slot combinations from the Home Assistant
[intent metadata][intents] are covered, including on/off (lights, switches,
covers, locks), brightness/color, climate, media transport and volume, timers,
todo lists, weather, date/time, and state queries.

A few intents are deliberately not exposed: the shopping-list intents (use a
todo list named "Shopping List" instead), broadcast/respond (they misfire on
declarative speech), vacuum/lawn-mower control, timer targeting by original
duration ("the 10 minute timer"), and deferred commands ("turn off the lights
in 10 minutes").

## Options

- `hf_token`: Hugging Face API token used for model downloads.
- `debug_logging`: enable verbose logging.

## Performance notes

- Latency is generation-bound: expect roughly 5–6 seconds per utterance on a
  Raspberry Pi 5 (4 threads), and 1–2 seconds on a typical x86 mini PC.
- The model runs entirely on CPU; ~8 GB of RAM is recommended.

[intents]: https://developers.home-assistant.io/docs/intent_builtin/
[gemma4]: https://deepmind.google/models/gemma/gemma-4/
[llama.cpp]: https://github.com/ggml-org/llama.cpp
[Wyoming]: https://github.com/OHF-Voice/wyoming
[voice]: https://www.home-assistant.io/voice_control/
