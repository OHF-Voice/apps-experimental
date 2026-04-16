# Sentence Transformers

Uses [sentence transformers][sentence_transformers] to match voice commands to [intents][] or [actions][].

<!-- Links -->
[sentence_transformers]: https://huggingface.co/sentence-transformers
[actions]: https://www.home-assistant.io/docs/scripts/perform-actions/
[intents]: https://developers.home-assistant.io/docs/intent_builtin/

- Timers
    - Start (`HassStartTimer`)
        - `seconds`
        - `minutes`
        - `hours`
    - Pause (`HassPauseTimer`)
    - Resume (`HassUnpauseTimer`)
    - Cancel (`HassCancelTimer`, `HassCancelAllTimers`)
    - Status (`HassTimerStatus`)
- Lights
    - On (`HassTurnOn`)
        - current area
        - `area`
    - Off (`HassTurnOff`)
        - current area
        - `area`
    - Brightness (`HassLightSet`)
        - current area
        - `area`
        - percentage
- Devices
    - On (`HassTurnOn`)
        - `name`
    - Off (`HassTurnOff`)
        - `name`
- Media control
    - Volume (`HassSetVolume`, `HassSetVolumeRelative`)
        - current area
        - `name`
        - level
        - up [percentage]
        - down [percentage]
    - Pause (`HassMediaPause`)
    - Resume (`HassMediaUnpause`)
    - Next (`HassMediaNext`)
    - Previous (`HassMediaPrevious`)

Play music - artist/album/track/genre

Weather forecast

Activate scene by name

Get area or house temperature

Get sensor value by name

Get state of a binary sensor - open/closed/locked/unlocked/etc.

Date and time
