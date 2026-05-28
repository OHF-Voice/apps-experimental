# Gemma 4 Script Runner

A [conversation][] agent built with [Gemma 4][gemma4] that runs Home Assistant [scripts][] that have been [expose][expose] to voice.

This app is a work in progress, so expect things to be unpolished!

## LLM

Gemma 4 is run on the CPU using [llama.cpp][] and a [quantized version][] of the [official model][]. Changing the quantization level, such as from Q5 to Q8, will change the accuracy, speed, and RAM usage of the agent.

The default model is a 5-bit (Q5) version:

- hf_repo: `bartowski/google_gemma-4-E2B-it-GGUF`
- hf_filename: `google_gemma-4-E2B-it-Q5_K_M.gguf`

If you'd like to try the higher-precision official Q8 quantization, use these settings:

- hf_repo: `ggml-org/gemma-4-E2B-it-GGUF`
- hf_filename: `gemma-4-E2B-it-Q8_0.gguf`

## Installation and first boot

Installing the app can take quite a while, since it builds an optimized version of [llama.cpp] for your CPU.

On first boot of the app, [Gemma 4][gemma4] is downloaded (about 4GB). If you have a Hugging Face account, putting your token in the app settings (`hf_token`) before starting the app may speed up the download.

Once the app boots, check "Settings -> Devices & services" for a newly discovered voice conversation agent and add it. Select this agent (`script-gemma4`) in your voice pipeline, optionally checking "Prefer handling commands locally" if you want Home Assistant to try to recognize commands before sending them to the LLM.

## Scripts and selectors

Create scripts and [expose][] them to voice. You **must** also expose any entities that you want to be able to refer to by name. After any changes are made, restart the app.

Give your scripts descriptive names and consider adding a description with more details. Add [fields][] to have Gemma 4 pass variables to your script. Make sure to add descriptions!

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

A special `satellite` variable is passed to each script with information about the [voice satellite][] that initiated the command. This variable has the following structure:

- `satellite.entity_id` - entity id of the [voice satellite][]
    - Useful for [responding][announce] back with a message
- `satellite.area_id` - id of the [area][] where the satellite is located
    - Useful for commands that target the current area
- `satellite.device_id` - id of the satellite's device (usually an [ESPHome][esphome] device)
    - Useful if you want to play something on the media player associated with the satellite

## Multiple commands

Gemma 4 can recognize and run multiple scripts. This works best with larger models, such as the official Q8 version (see above).

## State caching

To keep the speed reasonable, the agent caches the LLM state on startup whenever the scripts or [exposed][expose] entities have changed. Rebuilding the cached state can take several minutes.

## Tool call caching

If a sentence has been previously recognized, its result will be cached and the LLM will be skipped next time. The number of cached sentences is controlled by `tool_call_cache_size` (default: 100). The cache is cleared when the app restarts.

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
[selectors]: https://www.home-assistant.io/docs/blueprint/selectors/
[aliases]: https://www.home-assistant.io/voice_control/aliases/
[domain filter]: https://www.home-assistant.io/docs/blueprint/selectors/#domain
[llama.cpp]: https://github.com/ggml-org/llama.cpp
[quantized version]: https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF
[official model]: https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
