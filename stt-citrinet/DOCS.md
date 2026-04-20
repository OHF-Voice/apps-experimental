# Citrinet Speech-to-Text

Limited [Wyoming][wyoming] speech-to-text ([STT][stt]) app for [NeMO][nemo] CTC models.

This app will only recognize the voice commands you configure.

## Supported Languages

- `be` - Belarusian
- `ca` - Catalan
- `de` - German
- `en` - English
- `eo` - Esperanto
- `es` - Spanish
- `fr` - French
- `hi` - Hindi
- `hr` - Croatian
- `it` - Italian
- `mr` - Marathi
- `ru` - Russian
- `rw` - Kinyarwanda
- `zh` - Chinese

## Installation

Add the [OHF experimental][ohf-experimental] repo to your app repositories and install the `stt-citrinet` app.

Make sure to select a model from the app's configuration page.
Models are named `stt_<language>_<type>_<size>` with the "type" usually as conformer or citrinet.
Prefer conformer models for accuracy, but citrinet models for speed.

After installing, add the discovered `stt-citrinet` service for the [Wyoming][wyoming] integration.

## Usage

After starting the app, visit the web UI and add your voice commands.
Click "Save" to save and retrain.

Add one sentence per line with no punctuation (lower case).

### Template Syntax

The voice command file supports template syntax for more flexible command definitions:

- **Abbreviations**: Write out as words (e.g., `A.C.` → `a c`)
- **Optional text**: Surround with square brackets like `light[s]`
- **Alternative text**: Surround with parentheses like `(red|green|blue)`
- **Numbers**: Surround with curly braces:
  - Individual: `{1,2,3}`
  - Range: `{1..10}`
  - Range with step: `{0..100/10}`
  - Combined: `{1,2,3,5..25/5}`

See the in-app help for more examples.

[wyoming]: https://www.home-assistant.io/integrations/wyoming/
[stt]: https://www.home-assistant.io/integrations/stt/
[nemo]: https://docs.nvidia.com/nemo-framework/index.html
[ohf-experimental]: https://github.com/OHF-Voice/apps-experimental
