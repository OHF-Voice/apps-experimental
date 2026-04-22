# OHF Apps Experimental

Experimental voice apps (formally add-ons) for Home Assistant.

Expect things to be unpolished!

## Speech-to-Text

- [stt-canary](stt-canary/DOCS.md) - [speech-to-text][stt] app for NeMo's [Canary][canary].
    - Very large model that requires lots of RAM or a GPU
- [stt-qwen](stt-qwen/DOCS.md) - [speech-to-text][stt] app for [Qwen ASR][qwen-asr].
    - Medium-sized model that may be slow on older hardware
- [stt-citrinet](stt-citrinet/DOCS.md) - Limited [speech-to-text][stt] app for [NeMO][nemo] CTC models.
    - Fast and accurate recognition of the configured voice commands only
- [stt-coqui](stt-coqui/DOCS.md) - Limited [speech-to-text][stt] app for [Coqui STT][coqui-stt].
    - Fast and moderately accurate recognition of the configured voice commands only

## Intent Recognition

- [intent-sentence-transformers](intent-sentence-transformers/DOCS.md) - [conversation agent][conversation] app using [sentence transformers][sentence_transformers]
    - Flexibly match voice commands with intents and actions

## Wake Word Detection

- [wake-phonmatchnet](wake-phonmatchnet/DOCS.md) - [wake word][wake_word] app using [PhonMatchNet][phonmatchnet]
    - Streaming detection of arbitrary wake words

<!-- Links -->
[stt]: https://www.home-assistant.io/integrations/stt/
[canary]: https://huggingface.co/nvidia/canary-1b-v2
[coqui-stt]: https://github.com/coqui-ai/STT
[nemo]: https://docs.nvidia.com/nemo-framework/index.html
[qwen-asr]: https://github.com/QwenLM/Qwen3-ASR/
[sentence_transformers]: https://huggingface.co/sentence-transformers
[conversation]: https://www.home-assistant.io/integrations/conversation/
[wake_word]: https://www.home-assistant.io/integrations/wake_word/
[phonmatchnet]: https://github.com/ncsoft/PhonMatchNet
