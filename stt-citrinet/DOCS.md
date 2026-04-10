# Citrinet Speech-to-Text

Limited [Wyoming][wyoming] speech-to-text ([STT][stt]) app for [NeMO][nemo] CTC models.

This app will only recognize the voice commands you give it.

## Installation

Add the [OHF experimental][ohf-experimental] repo to your app repositories and install the `stt-citrinet` app.

After installing, add the discovered `stt-citrinet` service for the [Wyoming][wyoming] integration.

## Usage

After starting the app, visit the web UI and add your voice commands.
Click "Save" to save and retrain.

[wyoming]: https://www.home-assistant.io/integrations/wyoming/
[stt]: https://www.home-assistant.io/integrations/stt/
[nemo]: https://docs.nvidia.com/nemo-framework/index.html
[ohf-experimental]: https://github.com/OHF-Voice/apps-experimental
