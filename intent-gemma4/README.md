# Gemma 4 Intent Recognizer

Recognizes [Home Assistant intents][intents] locally with [Gemma 4][gemma4] and
[llama.cpp][], and emits them as [Wyoming][] `Intent` events.

Intent recognition is recast as LLM tool calling: one small tool per intent is
auto-generated from the Home Assistant intent metadata and aggressively
filtered by the exposed entities, areas, and floors of your home, so the model
only ever sees names it can actually target. A persisted llama.cpp prefix
cache keeps per-utterance latency low on CPU-only hardware such as a
Raspberry Pi 5.

The web UI includes a sentence tester (type a command, see the recognized
intent and slots), a listing of the exposed entity/area/floor names, a tools
override editor, and a latency + correctness benchmark.

See [DOCS.md](DOCS.md) for details.

<!-- Links -->
[intents]: https://developers.home-assistant.io/docs/intent_builtin/
[gemma4]: https://deepmind.google/models/gemma/gemma-4/
[llama.cpp]: https://github.com/ggml-org/llama.cpp
[Wyoming]: https://github.com/OHF-Voice/wyoming
