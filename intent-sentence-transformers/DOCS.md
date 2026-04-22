# Sentence Transformers Agent

Uses [sentence transformers][sentence_transformers] to match voice commands to [intents][] or [actions][].

## Usage

After installing the app, add "intent-sentence-transformers" in "Settings -> Devices & Services" ([Wyoming][wyoming] integration).

In the app's configuration, click "Open Web UI" and add your commands. Make sure to click "Save" to save and re-train.

Select `intent-sentence-transformers` as the conversation agent component in your [voice assistant pipeline](voice_control).

## Models

For English, the default model is [intfloat/e5-small-v2][en_model].

For other languages, the default model is [intfloat/multilingual-e5-small][multilingual_model].

You can override this in the app by changing the "model" configuration parameter to a different HuggingFace model id.

## Format

```yaml
# Language code used for number/duration parsing.
# See: https://github.com/MycroftAI/lingua-franca
language: "en"

# List of commands.
# A command can return and intent for Home Assistant to handle, or run a Home Assistant action.
commands:

  # Command with an intent
  - id: "command_1"
    description: "Example command 1"
    # These sentences are embedded with the sentence transformers model.
    sentences:
      - "list of sentences"
      - "that will match this command"
      
    intent: "MyIntent"
    # An intent can also have inferred slots:
    # intent:
    #   name: "MyIntent"
    #   slots:
    #     slot_name: "slot_value"
    
    # Command must have an implicit area from context.
    # This is usually the area of the voice satellite device.
    current_area: true
    # The "area" slot will contain the implicit area id.
    # You can override this.
    # current_area:
    #   slot: "different_slot"

    # Command text is expected to contain a duration like "5 minutes".
    # This is parsed with lingua-franca (https://github.com/MycroftAI/lingua-franca).
    duration: true
    # The duration is split in "seconds", "minutes", and "hours" slots.
    # You can override this.
    # duration:
    #   seconds_slot: "different_seconds_slot"
    #   minutes_slot: "different_minutes_slot"
    #   hours_slot: "different_hours_slot"
    
    # Command text is expected to contain a number as either digits or words.
    # This is parsed with lingua-franca (https://github.com/MycroftAI/lingua-franca).
    # The range is not currently validated.
    percentage:
      slot: "slot_for_number"

    # jinja2 template that's rendered locally.
    # The matched slot values are available as "slots.<slot_name>".
    response: "Response 1 with {{ slots.my_slot }}"
    
    # The response can also be rendered remotely in Home Assistant.
    # hass_response: "{{ ... }}"
    
  # Command with an action
  - id: "command_2"
    description: "Example command 2"
    sentences:
      - "sentences for action command"
      
    action:
      # Home Assistant action
      action: "domain.service"
      
      # Target for action (optional).
      # The values are rendered as templates in Home Assistant.
      target:
        # May be one id or a list
        entity_id: "target_entity_id"
        area_id: "target_area_id"
        floor_id: "target_floor_id"
        
      # Extra data for action (optional).
      # The values are rendered as templates in Home Assistant.
      data:
        key: "value"
    
```

### Variables

The following variables are available in templates:

- `slots` - object with inferred and matched slots
- `satellite` - available if `satellite_id` is provided during recognition
    - `entity_id` - entity id of [Assist satellite][assist_satellite]
    - `device_id` - id of satellite device
    - `area_id` - area id of satellite device
    - `area_name` - name of the satellite device's area
    - `floor_id` - floor id of satellite device area
    
### Lists

```yaml
  - id: "Turn on by name"
    action:
      action: "light.turn_on"
      target:
        entity_id: "{{ lists.name }}"
    description: "test command"
    sentences:
      - "turn on {name}"
    sentence_lists:
      name:
        - in: light 1
          out: light.bed_light
        - in: light 2
          out: light.office_rgbw_lights
    response: "Turned on {{ lists.name.text }}"

  - id: "Turn off by name"
    action:
      action: "light.turn_off"
      target:
        entity_id: "{{ lists.name }}"
    description: "test command"
    sentences:
      - "turn off {name}"
    sentence_lists:
      name:
        - in: light 1
          out: light.bed_light
        - in: light 2
          out: light.office_rgbw_lights
    response: "Turned off {{ lists.name.text }}"
```

## Example

Full example of commands:

```yaml
---
language: en

settings:
  thermostat_id: "sensor.thermostat_temperature"
  front_door_id: "lock.deadbolt"
  garage_door_id: "binary_sensor.garage_door"
  weather_id: "weather.home"
  music_script_id: "script.play_music"
  calendar_id: "calendar.reminders"
  outside_temp_id: "sensor.gw2000b_feels_like_temperature"
  scene_id: "scene.party_time"

errors:
  unknown_command: |
    "Sorry, I couldn't understand the command: {{ text }}."

commands:
  # ---------------------------------------------------------------------------
  # Date/Time
  # ---------------------------------------------------------------------------
  - id: "current_date"
    intent: HassGetCurrentDate
    description: "get current date"
    sentences:
      - "what's the date"
    threshold: 0.85
    hass_response: |
      {% set day = now().day %}
      {% set suffix = 'th' %}
      {% if day % 100 not in [11, 12, 13] %}
        {% if day % 10 == 1 %}
          {% set suffix = 'st' %}
        {% elif day % 10 == 2 %}
          {% set suffix = 'nd' %}
        {% elif day % 10 == 3 %}
          {% set suffix = 'rd' %}
        {% endif %}
      {% endif %}

      {{ now().strftime("%B ") }}{{ day }}{{ suffix }}, {{ now().year }}

  - id: "current_time"
    intent: HassGetCurrentTime
    description: "get current time"
    threshold: 0.85
    sentences:
      - "what time is it"
    hass_response: |
      {{ now().strftime("%I")|int }}:{{ now().strftime("%M %p") }}

  # ---------------------------------------------------------------------------
  # Timers
  # ---------------------------------------------------------------------------
  - id: "start_timer"
    intent: HassStartTimer
    description: "start a timer with a duration"
    duration: true
    sentences:
      - "set a timer for 10 seconds"
      - "10 minute timer"
      - "start timer for ten hours"
    response: |
      {% set hours = slots.hours | default(0) | int %}
      {% set minutes = slots.minutes | default(0) | int %}
      {% set seconds = slots.seconds | default(0) | int %}
      {% set parts = [] %}

      {% if hours > 0 %}
      {% set parts = parts + [hours ~ " hour" ~ ("s" if hours != 1 else "")] %}
      {% endif %}

      {% if minutes > 0 %}
      {% set parts = parts + [minutes ~ " minute" ~ ("s" if minutes != 1 else "")] %}
      {% endif %}

      {% if seconds > 0 or parts | length == 0 %}
      {% set parts = parts + [seconds ~ " second" ~ ("s" if seconds != 1 else "")] %}
      {% endif %}

      Starting timer for {{ parts | join(", ") }}.

  - id: "pause_timer"
    intent: HassPauseTimer
    description: "pause the latest timer"
    sentences:
      - "pause timer"
    response: "Pausing timer."

  - id: "unpause_timer"
    intent: HassUnpauseTimer
    description: "resume the latest timer"
    sentences:
      - "resume timer"
    response: "Resuming timer."

  - id: "cancel_timer"
    intent: HassCancelTimer
    description: "cancel the latest timer"
    sentences:
      - "cancel timer"
    response: "Canceling timer."

  - id: "cancel_all_timers"
    intent: HassCancelAllTimers
    description: "cancel all timers"
    sentences:
      - "cancel all timers"
    response: "Canceling all timers."

  # ---------------------------------------------------------------------------
  # Lights
  # ---------------------------------------------------------------------------
  - id: "turn_on_lights"
    intent:
      name: HassTurnOn
      slots:
        domain: light
    description: "turn on the lights in the current area"
    current_area: true
    sentences:
      - "turn on the lights"
    response: "Turning on the lights in the {{ satellite.area_name }}."

  - id: "turn_off_lights"
    intent:
      name: HassTurnOff
      slots:
        domain: light
    description: "turn off the lights in the current area"
    current_area: true
    sentences:
      - "turn off the lights"
    response: "Turning off the lights in the {{ satellite.area_name }}."

  - id: "set_brightness_lights"
    intent: HassLightSet
    description: "set the brightness of lights in the current area"
    current_area: true
    percentage:
      slot: brightness
    sentences:
      - "set brightness to 10 percent"
    response: |
      Setting brightness of lights in the {{ satellite.area_name }} to {{ slots.brightness }} percent.

  # ---------------------------------------------------------------------------
  # Media
  # ---------------------------------------------------------------------------
  - id: "media_pause"
    intent: HassMediaPause
    description: "pauses playing media player in current area"
    current_area: true
    sentences:
      - "pause"
      - "pause the music"
      - "pause TV"
    response: "Pausing media."

  - id: "media_unpause"
    intent: HassMediaUnpause
    description: "resumes playing media player in current area"
    current_area: true
    sentences:
      - "resume"
      - "resume the music"
      - "unpause music"
    response: "Resuming media."

  - id: "media_next"
    intent: HassMediaNext
    description: "skips to next track on playing media player in current area"
    current_area: true
    sentences:
      - "next"
      - "skip to the next track"
      - "skip this song"
    response: "Skipping to next track."

  - id: "media_volume_up"
    intent:
      name: HassSetVolumeRelative
      slots:
        volume_step: "up"
    description: "increases the volume of the playing media player in current area"
    current_area: true
    sentences:
      - "volume up"
      - "turn up the volume"
      - "make the music louder"
    response: "Increasing volume."

  - id: "media_volume_down"
    intent:
      name: HassSetVolumeRelative
      slots:
        volume_step: "down"
    description: "decreases the volume of the playing media player in current area"
    current_area: true
    sentences:
      - "volume down"
      - "turn down the volume"
      - "make the music quieter"
    response: "Decreasing volume."

  - id: "media_volume_set"
    intent: HassSetVolume
    description: "decreases the volume of the playing media player in current area"
    current_area: true
    percentage:
      slot: "volume_level"
    sentences:
      - "set volume to 10 percent"
      - "set music volume to fifty percent"
      - "volume 10%"
    response: "Setting volume to {{ slots.volume_level }} percent."

  # ---------------------------------------------------------------------------
  # Weather
  # ---------------------------------------------------------------------------
  - id: "weather_forecast"
    description: "gets the current weather forecast"
    sentences:
      - what is the weather today
      - tell me the weather forecast
      - how's the weather
    hass_response: |
      {% set weather_id = settings.weather_id %}
      {% if weather_id %}
        {% set name = state_attr(weather_id, 'friendly_name') or 'Outside' %}
        {% set cond = states(weather_id) %}
        {% set temp = state_attr(weather_id, 'temperature') %}
        {% set unit = state_attr(weather_id, 'temperature_unit') or '°' %}
        {% set hum = state_attr(weather_id, 'humidity') %}
        {% set wind = state_attr(weather_id, 'wind_speed') %}
        {% set wind_unit = state_attr(weather_id, 'wind_speed_unit') %}

        {% macro pretty_condition(c) -%}
          {{ (c or 'unknown') | replace('_', ' ') }}
        {%- endmacro %}

        Currently {{ pretty_condition(cond) }}{% if temp is not none %}, {{ temp | round(0) }}{{ unit }}{% endif %}{% if hum is not none %}, humidity {{ hum }} percent{% endif %}{% if wind is not none %}, wind {{ wind | round(0) }}{% if wind_unit %} {{ wind_unit }}{% endif %}{% endif %}.

      {% else %}
      No weather entity set.
      {% end

  # ---------------------------------------------------------------------------
  # Temperature
  # ---------------------------------------------------------------------------
  - id: "temperature_inside"
    description: "gets the temperature inside the house"
    sentences:
      - what is the temperature
      - what's the temperature in the house
    hass_response: >-
      {% set thermostat_id = settings.get('thermostat_id') %}
      {% if thermostat_id %}
        {% set temp = states(thermostat_id) | float %}
        {{ temp|int if temp == temp|int else temp }} degrees
      {% else %}
        No thermostat is set.
      {% endif %}

  - id: "temperature_outside"
    description: "gets the temperature outside the house"
    sentences:
      - what is the temperature outside
      - how hot is it
      - how cold is it outside
    hass_response: >-
      {% set outside_temp_id = settings.get('outside_temp_id') %}
      {% if outside_temp_id %}
        {% set temp = states(outside_temp_id) | float %}
        {{ temp|int if temp == temp|int else temp }} degrees
      {% else %}
        No temperature sensor is set.
      {% endif %}

  # ---------------------------------------------------------------------------
  # Doors
  # ---------------------------------------------------------------------------
  - id: "front_door_locked"
    description: "tells if the front door is locked or unlocked"
    sentences:
      - is the front door locked
      - is the front door unlocked
    hass_response: >-
      {% set front_door_id = settings.get('front_door_id') %}
      {% if front_door_id %}
        {% set state = states(front_door_id) %}
        {% if state == 'locked' %}
        Front door is locked.
        {% else %}
        Front door is currently unlocked.
        {% endif %}
      {% else %}
        No front door is set.
      {% endif %}

  - id: "garage_door_open"
    description: "tells if the garage door is open or closed"
    sentences:
      - is the garage door open
      - is the garage door closed
    hass_response: >-
      {% set garage_door_id = settings.get('garage_door_id') %}
      {% if garage_door_id %}
        {% set state = states(garage_door_id) %}
        {% if state == 'on' %}
        Garage door is currently open.
        {% else %}
        Garage door is closed.
        {% endif %}
      {% else %}
        No garage door is set.
      {% endif %}

  # ---------------------------------------------------------------------------
  # Music
  # ---------------------------------------------------------------------------
  - id: "play_music"
    action:
      action: "script.turn_on"
      target:
        entity_id: "{{ settings.music_script_id }}"
      data:
        variables:
          search_query: "{{ slots.search_query }}"
          area_id: "{{ satellite.area_id }}"
    description: "plays an album by an artist"
    patterns:
      - "play (?P<search_query>.+)"
    response: "Playing {{ slots.search_query }}"

  # ---------------------------------------------------------------------------
  # Todo
  # ---------------------------------------------------------------------------
  - id: "add_todo"
    intent: HassListAddItem
    description: "add an item to a todo list"
    templates:
      - "add {item} to [the|my] {name} [list]"
    response: "Adding {{ slots.item }} to {{ slots.name }}"

  - id: "complete_todo"
    intent: HassListCompleteItem
    description: "complete an item on a todo list"
    templates:
      - "(complete|check off) {item} from [the|my] {name} [list]"
    response: "Checking off {{ slots.item }} from {{ slots.name }}"

  # ---------------------------------------------------------------------------
  # Reminders
  # ---------------------------------------------------------------------------
  - id: "add_reminder"
    action:
      action: calendar.create_event
      target:
        entity_id: "{{ settings.calendar_id }}"
      data:
        summary: "{{ slots.task }}"
        start_date_time: |
          {{ extract_datetime(slots.duration).isoformat() }}
        end_date_time: |
          {{ (extract_datetime(slots.duration) + timedelta(minutes=5)).isoformat() }}
    description: "add a reminder event to a calendar"
    templates:
      - "remind me in {duration} to {task}"
      - "remind me to {task} in {duration}"
    response: "Adding reminder to {{ slots.task }} in {{ slots.duration }}"

  # ---------------------------------------------------------------------------
  # Scenes
  # ---------------------------------------------------------------------------
  - id: "activate_scene"
    action:
      action: "scene.turn_on"
      target:
        entity_id: "{{ settings.scene_id }}"
    description: "activates a scene"
    sentences:
      - "party time"
      - "activate party time scene"
    response: "It's party time! Excellent!"

  # ---------------------------------------------------------------------------
  # Misc
  # ---------------------------------------------------------------------------
  - id: "nevermind"
    intent: HassNevermind
    description: "cancels command"
    sentences:
      - "nevermind"
    response: ""
```
    
## Future Ideas

- Different response for when intent or action fails
- Better distance metric than just cosine
- Lists should be available to templates too
- Add webhooks alongside intents and actions
- Allow for multiple sentence files and languages
    
<!-- Links -->
[sentence_transformers]: https://huggingface.co/sentence-transformers
[actions]: https://www.home-assistant.io/docs/scripts/perform-actions/
[intents]: https://developers.home-assistant.io/docs/intent_builtin/
[assist_satellite]: https://www.home-assistant.io/integrations/assist_satellite/
[voice_control]: https://www.home-assistant.io/voice_control/
[en_model]: https://huggingface.co/intfloat/e5-small-v2
[multilingual_model]: https://huggingface.co/intfloat/multilingual-e5-small
