# Gemma 4 Script Runner

A [conversation][] agent built with [Gemma 4][gemma4] that runs Home Assistant
[scripts][] which have been [exposed][expose] to voice.

This app is experimental: it only runs scripts you write for it (see the
[blueprints][] for examples), and response times depend heavily on your
hardware — see [Benchmarks](#benchmarks) below.

![architecture](https://raw.githubusercontent.com/OHF-Voice/apps-experimental/main/script-gemma4/res/architecture.png)

## Getting started

To install the app, first add `https://github.com/OHF-Voice/apps-experimental`
to your app repositories by:

1. Go to "Settings -> Apps"
2. Click the "Install app" button
3. In the 3-dot menu, click "Repositories"
4. Click the "Add" button

After adding the repository, make sure you're in "Settings -> Apps" and "Install
apps". You may need to click "Check for updates" from the 3-dot menu.

Scroll down and find "OHF Experimental Apps" and choose "Gemma 4 Script Runner",
then click "Install". To check on installation progress, go to "Settings ->
System -> Logs" and change to "Supervisor".

### First boot

Installing the app can take quite a while, since it builds an optimized version
of [llama.cpp][] for your CPU.

On the first boot, [Gemma 4][gemma4] is downloaded (about 4GB) and the cached
model state is built. This can take several minutes. Setting `hf_token` to a
Hugging Face API token before starting the app makes the download faster and
more reliable.

The web UI is available while this happens and shows a "model is still loading"
notice; voice commands are not handled until it is ready. Once the app is ready,
check "Settings -> Devices & services" for a newly discovered [wyoming][]
conversation agent called "Gemma 4 Script Runner" and add it. Select this agent
(`script-gemma4`) in your voice pipeline, optionally checking "Prefer handling
commands locally" if you want Home Assistant to try to recognize commands before
sending them to the LLM.

## Options

- `hf_repo`: Hugging Face repository id for the Gemma 4 GGUF model.
- `hf_model`: model filename within `hf_repo` (see [LLM](#llm) below).
- `hf_token`: Hugging Face API token, used for faster model downloads.
- `tool_call_cache_size`: number of recognized sentences to remember (default:
  `100`, see [Tool call caching](#tool-call-caching)).
- `n_ctx`: model context size in tokens (default: `0`, which sizes it
  automatically — see [Context size](#context-size)).
- `n_ctx_overhead`: tokens to reserve beyond the system prompt when `n_ctx` is
  automatic (default: `128`).
- `n_threads`: CPU threads for the model (default: `0`, llama.cpp's own default
  — see [CPU threads](#cpu-threads)).
- `max_tokens`: maximum tokens the model may generate per command (default:
  `64`). Raise this only if you expect many tool calls at once, since it also
  raises the automatic context size.
- `debug_logging`: enable verbose logging, including llama.cpp's own output.

Changes to these options take effect on restart.

## LLM

Gemma 4 is run on the CPU using [llama.cpp][] and a [quantized version][] of the
[official model][]. Changing the quantization level, such as from Q5 to Q8, will
change the accuracy, speed, and RAM usage of the agent.

The default model is a 5-bit (Q5) version:

- `hf_repo`: `bartowski/google_gemma-4-E2B-it-GGUF`
- `hf_model`: `google_gemma-4-E2B-it-Q5_K_M.gguf`

If you'd like to try the higher-precision official Q8 quantization, use these
settings:

- `hf_repo`: `ggml-org/gemma-4-E2B-it-GGUF`
- `hf_model`: `gemma-4-E2B-it-Q8_0.gguf`

### Context size

The size of the [llama.cpp][] context (`n_ctx`) is automatically determined based
on the size of the generated tools. If your scripts have large lists of entities
and areas, a larger context size will be needed and increase app RAM usage.
Decreasing the number of [exposed][expose] entities can help keep the context
size small.

### CPU threads

The number of CPU threads used by the model is set with `n_threads` (default:
`0`, which uses llama.cpp's own default). Generation speed is limited by memory
bandwidth rather than CPU, so adding threads past a few cores gives little or no
speedup. Because the model's threads synchronize on every token, a thread that
gets descheduled by other work (such as Home Assistant running on the same
machine) stalls the rest, causing latency spikes under load. On a box shared
with Home Assistant, leave a core or two free (for example, set `n_threads` to
`2` on a 4-core device) for more consistent response times.

## Scripts and selectors

Create scripts and [expose][] them to voice by "More Info -> Settings -> Voice
assistants" and clicking "Expose". You **must** also expose any entities that
you want to be able to refer to by name (or [alias][aliases]). After any changes
are made, restart the app.

Give your scripts descriptive names and consider adding a description with more
details. Add [fields][] to have Gemma 4 pass variables to your script. Make sure
to add descriptions!

See the [blueprints][] for example scripts.

The following field [selectors][] are supported:

- Area
    - Uses all available area names and [aliases][]
- Boolean
- Color temperature
- Date
- Date & time
- Duration
- Entity
    - Uses all [exposed][expose] entity names
    - Add a [domain filter][] to restrict possible entities
- Floor
    - Uses all available floor names and [aliases][]
- Number
    - Set min/max if it makes sense
- RGB color
- Select
- Text
- Time

### satellite variable

A special `satellite` variable is passed to each script with information about
the [voice satellite][] that initiated the command. This variable has the
following properties:

- `entity_id` - entity id of the [voice satellite][]
    - Useful for [responding][announce] back with a message
- `area_id` - id of the [area][] where the satellite is located
    - Useful for commands that target the current area
- `floor_id` - id of the [floor][] where the satellite is located
- `device_id` - id of the satellite's device (usually an [ESPHome][esphome] device)
    - Useful if you want to play something on the media player associated with the satellite
- `media_player_id` - id of the closest [media player][]
    - Search order for media players is satellite device, satellite area, and satellite floor
- `music_player_id` - id of the closest [media player][] that supports [Music Assistant][]
    - Search order for music players is satellite device, satellite area, and satellite floor
- `music_assistant_id` - config entry id of [Music Assistant][]
    - For calling actions like `music_assistant.search`
- `language` - language code of the input text
    - May be something like `de`, `en_GB`, or `pt-BR`

## Multiple commands

Gemma 4 can recognize and run multiple scripts, for example "turn on the lights
and play The Beatles". This works best with larger models, such as the official
Q8 version (see above). Make sure to write your scripts so that more than one
could run at a time!

## State caching

To keep the speed reasonable, the agent caches the LLM state on startup whenever
the scripts or [exposed][expose] entities have changed. Rebuilding the cached
state can take several minutes.

## Tool call caching

If a sentence has been previously recognized, its result will be cached and the
LLM will be skipped next time. The number of cached sentences is controlled by
`tool_call_cache_size` (default: 100). The cache is cleared when the app
restarts.

## Web UI

A small web interface is available that shows the generated LLM tools (OpenAI
function spec). This is for debugging only.

A benchmark page is available at `/benchmark` (relative to the ingress URL). It
runs a fixed fixture of tools and sentences (`benchmark.yaml`) against the loaded
model and reports per-sentence latency, throughput (tokens/sec), and correctness
(the parsed tool call vs. the expected one). The fixture is independent of your
exposed scripts so results are reproducible and comparable across machines and
model/config changes. The live model state is snapshotted and restored, so
running a benchmark does not disturb the assistant &mdash; but voice handling
pauses for its duration. While the model is still loading the page says so and
the button stays disabled.

## Benchmarks

Seconds per command with 5 scripts and 35 exposed entities.

- AMD Ryzen 9 5950X - 0.5-1.5 seconds
- Intel Core i5-4570T - 2-3 seconds
- Raspberry Pi 5 - 3-6 seconds
- Home Assistant Green - 15-20 seconds

<!-- Links -->
[conversation]: https://www.home-assistant.io/integrations/conversation/
[gemma4]: https://deepmind.google/models/gemma/gemma-4/
[scripts]: https://www.home-assistant.io/integrations/script/
[expose]: https://www.home-assistant.io/voice_control/voice_remote_expose_devices/
[fields]: https://www.home-assistant.io/integrations/script/#passing-variables-to-scripts
[voice satellite]: https://www.home-assistant.io/integrations/assist_satellite/
[esphome]: https://www.home-assistant.io/integrations/esphome
[announce]: https://www.home-assistant.io/integrations/assist_satellite/#action-announce
[area]: https://www.home-assistant.io/getting-started/concepts-terminology/#areas
[floor]: https://www.home-assistant.io/getting-started/concepts-terminology/#floors
[selectors]: https://www.home-assistant.io/docs/blueprint/selectors/
[aliases]: https://www.home-assistant.io/voice_control/aliases/
[domain filter]: https://www.home-assistant.io/docs/blueprint/selectors/#domain
[llama.cpp]: https://github.com/ggml-org/llama.cpp
[quantized version]: https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF
[official model]: https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
[media player]: https://www.home-assistant.io/integrations/media_player
[Music Assistant]: https://www.home-assistant.io/integrations/music_assistant/
[wyoming]: https://www.home-assistant.io/integrations/wyoming/
[blueprints]: https://github.com/OHF-Voice/apps-experimental/tree/main/script-gemma4/blueprints
