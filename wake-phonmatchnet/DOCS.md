# PhonMatchNet Wake Word

[Wyoming][wyoming] [wake word][wake_word] detection service using [PhonMatchNet](https://github.com/ncsoft/PhonMatchNet).

Uses a "universal" keyword search model to detect arbitrary English wake words.
Can detect multiple wake words simultaneously.

## Wake Words

Set the `wake_word` config value to one or more wake words, separated by commas.
For example: `okay nabu, hey jarvis` will detect both "okay nabu" and "hey jarvis."

Sometimes, it's best to spell out more clearly how a wake word sounds. In this case, you can separate the spoken (input) and phrase (output) forms with a colon.
For example: `glah-duhs:glados` will detect "glados" but spelled out more as its pronounced (hint: turn on debug logging to see the [ARPABet phonemes][arpabet] for each wake word).

## Sensitivity

Adjust `threshold` and `trigger_count` to change the sensitivity. The defaults are 0.5 and 1, respectively.

Increasing the threshold means the model must be more sure before triggering. Increasing the trigger count means that multiple audio windows in a row must exceed the threshold before triggering.

Sometimes it's better to have a lower threshold and a higher trigger count instead of just a high threshold. For example: `threshold` = 0.2 and `trigger_count` = 2.

## Window and Hop Lengths

The `window_length` and `hop_length` control how many seconds of audio are processed by the wake word model. A wake word must fit within `window_length`, but a longer window means more audio to process each time. Somewhere between 1.25 and 1.75 seconds seems to be ideal for most wake words.

The `hop_length` is how many seconds of audio are discarded to make room for the next window. A smaller `hop_length` means that there's more of a chance that the wake word will be correctly aligned within the window. But smaller hops means that many more windows much be processed, and will increase the detection delay. Anything below 0.5 seconds seems to be too small.

## Notes

False positives significantly increase with certain wake words, like "hey jarvis". Anything that is short and easily confused with other phrases increases the risk of false positives.

Automatic gain control (AGC) on the incoming audio should be turned **off** when possible. AGC increases the overall probabilities, and so increases false positives. Audio normalization is done automatically, so no gain boosting is required.

## Future Ideas

- Performance improvements - audio features are recomputed every window
- Allow phonemes directly for wake word - more control over pronunciations
- Allow different thresholds/trigger counts for each wake word
- Train model on more datasets - trained on LibriSpeech clean 360 + Qualcomm + Google keywords

[wyoming]: https://www.home-assistant.io/integrations/wyoming/
[wake_word]: https://www.home-assistant.io/integrations/wake_word/
[arpabet]: https://en.wikipedia.org/wiki/ARPABET
