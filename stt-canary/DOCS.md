# Canary Speech-to-Text

[Wyoming][wyoming] speech-to-text ([STT][stt]) app for NeMo's [Canary][canary].

**NOTE:** This model is quite large, so you will need a lot of RAM or a GPU to run it!

## Supported Languages

- bg - Bulgarian
- hr - Croatian
- cs - Czech
- da - Danish
- nl - Dutch
- en - English
- et - Estonian
- fi - Finnish
- fr - French
- de - German
- el - Greek
- hu - Hungarian
- it - Italian
- lv - Latvian
- lt - Lithuanian
- mt - Maltese
- pl - Polish
- pt - Portuguese
- ro - Romanian
- sk - Slovak
- sl - Slovenian
- es - Spanish
- sv - Swedish
- ru - Russian
- uk - Ukrainian

## Usage

After installing the app, add "stt-canary" in "Settings -> Devices & Services" ([Wyoming][wyoming] integration).

Select `stt-canary` as the speech-to-text component in your [voice assistant pipeline](voice_control).

[wyoming]: https://www.home-assistant.io/integrations/wyoming/
[stt]: https://www.home-assistant.io/integrations/stt/
[canary]: https://huggingface.co/nvidia/canary-1b-v2
[voice_control]: https://www.home-assistant.io/voice_control/
