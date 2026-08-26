# Changelog

## 1.7.0

- Update `llama-cpp-python` from 0.3.24 to 0.3.35 and invalidate saved model
  state across llama.cpp, model, context-size, and flash-attention changes
- Enable CPU flash attention by default. On a Raspberry Pi 5 it reduced median
  latency by about 3.6% and prefix rebuild time by about 3.9%, without adding a
  separate compilation step
- Do not run a script when the model omits one of its required fields, and drop
  fields that are not part of the script's schema
- Fix satellite lookup when the satellite entity has its own area or when its
  device id must be resolved from the entity registry
- Preserve per-script field names when global names are edited
- Apply Home Assistant reloads, targeting changes, and name changes
  transactionally, leaving the working model and UI state untouched if the new
  prefix cannot be built
- Wait for Home Assistant to accept script calls before reporting a command as
  handled, while allowing successful calls in a multi-command request to
  proceed when another call fails
- Recover automatically from corrupt or rejected saved model state, and restore
  the previous model even when a failed tool change attempted to grow context
- Accept single-quoted tool values containing commas and safely ignore malformed
  tool calls

## 1.6.0

- Use the voice satellite's area to pick between things that share a name. Home
  Assistant happily allows three media players all called "Media Player"; the
  one in the area the command came from now wins, then the one on that floor.
  Previously whichever entity happened to be built last silently won, making the
  others unreachable
- When that does not settle it, the field is dropped rather than guessed, and the
  tester reports which candidates it was torn between. Fields whose names are
  shared are flagged on the Scripts page
- Names can now be overridden for a single script's field, from an "Override
  names for this field" link next to that field. Useful when two things share a
  name inside one area, where the satellite cannot help
- The tester takes a "current area", so satellite-dependent behaviour can be
  tried without a satellite

## 1.5.0

- Add a **Names** page: change what the model may call each entity, area, and
  floor, independently of Home Assistant. Add a name people actually say, rename
  to tell apart entities that share a name (the model cannot target any of them
  reliably otherwise), or clear the names to keep something out of the enums
  entirely. Home Assistant is not modified
- Add **Reload from Home Assistant** to the Scripts page, so a newly exposed
  script, a rename, or a new alias can be picked up without restarting
- Name overrides are stored by Home Assistant id and only where they differ, and
  entries for entities, areas, or floors that no longer exist are dropped

## 1.4.0

- The web UI home page now lists every script in Home Assistant, split into the
  ones the model can call and the ones it cannot, with each script's fields and
  how many names each field can match
- Add a sentence tester: type a command and see which script it would run and
  with what variables, without running it
- Choose which scripts are targeted, independently of Home Assistant's voice
  exposure. Exposure remains the default; only the exceptions are stored (in
  `/data/overrides.yaml`), so newly exposed scripts are still picked up, and
  entries for deleted scripts are dropped automatically. Applying a change
  rebuilds the model prefix and grows the context if the new tool set needs it
- A script built from a blueprint reported no description, so the model was only
  told the script's *name*; it now inherits the blueprint's description
- Don't pass a name the model invented to a script as a null variable. Drop it
  instead, and don't run the script at all if it was a required field
- The "model loading" notice now covers building the model prefix, not just
  downloading the model

## 1.3.0

- Web UI and container health check now come up before the model loads, so the
  first boot no longer looks like a dead app while Gemma 4 downloads; the
  benchmark page reports "model is still loading" until it is ready
- Fix the `n_ctx`, `n_ctx_overhead`, and `max_tokens` options, which were
  accepted and documented but never passed to the agent
- `debug_logging` now also enables llama.cpp's own output
- Add a benchmark page (`/benchmark`) that runs a fixed tool/sentence fixture
  against the loaded model and reports correctness, latency, and tokens/second
- Fix the mapping of recognized tool calls back to script fields
- Fix the `hf_model` option showing as a raw key in the app configuration UI,
  and document every option

## 1.2.0

- Add `n_ctx`, `n_ctx_overhead`, `n_threads`, and `max_tokens` options
- Build llama.cpp in a separate stage so app rebuilds are faster
- Move model processing off the event loop onto a dedicated worker thread

## 1.1.0

- Add on/off script blueprints
- Split artist/album/track in the play music blueprint
- Pass the closest media player, music player, and Music Assistant config entry
  to scripts in the `satellite` variable

## 1.0.0

- Initial release
