# Changelog

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
