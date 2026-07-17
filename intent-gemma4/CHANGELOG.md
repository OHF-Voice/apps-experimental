# Changelog

## 1.0.0

- Initial release
- Recognizes Home Assistant intents locally with Gemma 4 (llama.cpp, CPU) and
  emits them as Wyoming `Intent` events
- Tools are auto-generated from the Home Assistant intent metadata and
  aggressively filtered by the exposed entities, areas, and floors (with
  aliases) of the connected home
- Persisted llama.cpp prefix cache: the system prompt + tools prefix is built
  once and restored on restart, so only the spoken sentence is evaluated per
  request
- Web UI: sentence tester with the recognized intent and slots, exposed
  entity/area/floor listing, tools YAML override editor, and a latency +
  correctness benchmark
- Manual tools override files saved by pre-release versions are detected and
  ignored (tools are auto-generated instead)
