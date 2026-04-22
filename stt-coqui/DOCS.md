# Coqui Speech-to-Text

Limited [Wyoming][wyoming] speech-to-text ([STT][stt]) app for [Coqui STT][coqui-stt].

This app will only recognize the voice commands you configure.

## Supported Languages

- `am_ET` - Amharic (Ethiopia)
- `br_FR` - Breton (France)
- `ca_ES` - Catalan (Spain)
- `cnh_MM` - Hakha Chin (Myanmar)
- `cs_CZ` - Czech (Czech Republic)
- `ctp_MX` - Western Highland Chatino (Mexico)
- `cv_RU` - Chuvash (Russia)
- `cy_GB` - Welsh (United Kingdom)
- `de_DE` - German (Germany)
- `dv_MV` - Dhivehi (Maldives)
- `el_GR` - Greek (Greece)
- `en_US` - English (United States)
- `es_ES` - Spanish (Spain)
- `et_EE` - Estonian (Estonia)
- `eu_ES` - Basque (Spain)
- `fa_IR` - Persian (Iran)
- `fi_FI` - Finnish (Finland)
- `fy_NL` - Western Frisian (Netherlands)
- `ga_IE` - Irish (Ireland)
- `hi_IN` - Hindi (India)
- `hsb_DE` - Upper Sorbian (Germany)
- `hu_HU` - Hungarian (Hungary)
- `id_ID` - Indonesian (Indonesia)
- `it_IT` - Italian (Italy)
- `ka_GE` - Georgian (Georgia)
- `kv_RU` - Komi-Zyrian (Russia)
- `ky_KG` - Kyrgyz (Kyrgyzstan)
- `lb_LU` - Luxembourgish (Luxembourg)
- `lg_UG` - Luganda (Uganda)
- `lt_LT` - Lithuanian (Lithuania)
- `lv_LV` - Latvian (Latvia)
- `mn_MN` - Mongolian (Mongolia)
- `mt_MT` - Maltese (Malta)
- `nl_NL` - Dutch (Netherlands)
- `or_IN` - Odia (India)
- `pl_PL` - Polish (Poland)
- `pt_PT` - Portuguese (Portugal)
- `rm_CH_sursilv` - Romansh Sursilvan (Switzerland)
- `rm_CH_vallader` - Romansh Vallader (Switzerland)
- `ro_RO` - Romanian (Romania)
- `ru_RU` - Russian (Russia)
- `rw_RW` - Kinyarwanda (Rwanda)
- `sah_RU` - Sakha (Russia)
- `sl_SI` - Slovenian (Slovenia)
- `sw_CD` - Swahili (Congo - DRC)
- `ta_IN` - Tamil (India)
- `th_TH` - Thai (Thailand)
- `tos_MX` - Upper Sierra Totonac (Mexico)
- `tr_TR` - Turkish (Turkey)
- `tt_RU` - Tatar (Russia)
- `uk_UA` - Ukrainian (Ukraine)
- `wo_SN` - Wolof (Senegal)
- `xty_MX` - Yoloxóchitl Mixtec (Mexico)
- `yo_NG` - Yoruba (Nigeria)

## Installation

Add the [OHF experimental][ohf-experimental] repo to your app repositories and install the `stt-coqui` app.

## Usage

After installing the app, add "stt-coqui" in "Settings -> Devices & Services" ([Wyoming][wyoming] integration).
Select `stt-coqui` as the speech-to-text component in your [voice assistant pipeline](voice_control).

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

[wyoming]: https://www.home-assistant.io/integrations/wyoming/
[stt]: https://www.home-assistant.io/integrations/stt/
[coqui-stt]: https://github.com/coqui-ai/STT
[voice_control]: https://www.home-assistant.io/voice_control/
