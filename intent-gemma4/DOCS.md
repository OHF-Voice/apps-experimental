# Gemma 4 Agent

[Conversation][conversation] agent that uses [Gemma 4][gemma4] with [llama.cpp][] exclusively for tool calling.

Tools and the [actions][] or [intents][] they execute in Home Assistant can be edited via a web UI.

## Building

This app uses [llama-cpp-python][] and builds it from source during installation in order to take advantage of all available CPU features. This can take a **long time**.

## Starting Up

When starting up, the model is downloaded, loaded into memory, and primed with a test prompt. This can take a **significant amount of time**, especially to download the model.

## Tools

Tools are defined using OpenAI's [function calling spec][tools].

## Commands

Each function name from a tool (under `tools:`) corresponds to command (under `commands:`).

Example:

```yaml
commands:
  - id: my_function
    response: "Response when function is called."
```

A command can execute an [action][actions] in Home Assistant. If the tool/function has arguments, they are accessed with `args`:

```yaml
commands:
  - id: my_function
    action:
      action: example.action
      data:
        data_arg: "{{ args.arg1 }}"
    response: "Response: {{ args.arg1 }}."
```

All templates (including the response) are [rendered inside Home Assistant][templates].

A command can also execute an [intent][intents] and pass slots:

```yaml
commands:
  - id: my_function
    intent:
      name: ExampleIntent
      slots:
        slot1: "{{ args.arg1 }}"
    response: "Response: {{ args.arg1 }}."
```

## Example Tools and Commands

```yaml
---
commands:
  - id: nevermind

  # date and time
  - id: current_date
    response: |
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
  - id: current_time
    response: |
      {{ now().strftime("%I")|int }}:{{ now().strftime("%M %p") }}

  # timers
  - id: set_timer
    intent:
      name: HassStartTimer
      slots:
        hours: "{{ args.total_seconds // 3600 }}"
        minutes: "{{ (args.total_seconds % 3600) // 60 }}"
        seconds: "{{ args.total_seconds % 60 }}"
        name: "{{ args.get('name') }}"
    response: >-
      {% set total_seconds = args.total_seconds | int %}
      {% set hours = total_seconds // 3600 %}
      {% set minutes = (total_seconds % 3600) // 60 %}
      {% set seconds = total_seconds % 60 %}
      {% set name = args.get('name') %}
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

      {% if name %}
      Starting a {{ parts | join(", ") }} timer for {{ name }}.
      {% else %}
      Starting timer for {{ parts | join(", ") }}.
      {% endif %}

  - id: control_timer
    intent:
      name: |
        {% if args.action == 'pause' %}
        HassPauseTimer
        {% elif args.action == 'resume' %}
        HassUnpauseTimer
        {% elif args.action == 'cancel' %}
        HassCancelTimer
        {% endif %}
    response: >-
      {% if args.action == 'pause' %}
      Pausing timer.
      {% elif args.action == 'resume' %}
      Resuming timer.
      {% elif args.action == 'cancel' %}
      Canceling timer.
      {% endif %}


  # lights
  - id: control_lights
    intent:
      name: |
        {% if args.action == 'on' %}
        HassTurnOn
        {% elif args.action == 'off' %}
        HassTurnOff
        {% endif %}
      slots:
        area: "{{ satellite.area_id }}"
        domain: light
    response: "Turning {{ args.action }} the lights in the {{ satellite.area_name }}."
  - id: set_brightness
    intent:
      name: HassLightSet
      slots:
        area: "{{ satellite.area_id }}"
        brightness: "{{ args.brightness }}"
    response: |
      Setting brightness of lights in the {{ satellite.area_name }} to {{ args.brightness }} percent.

  # weather and temperature
  - id: weather_forecast
    response: |
      {% set weather = states.weather | first %}
      {% set weather_id = weather.entity_id if weather else none %}
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
      {% endif %}
  - id: get_temperature
    response: >-
      {% set climate = states.climate | first %}
      {% set climate_id = climate.entity_id if climate else none %}
      {% if climate_id %}
        {% set temp = state_attr(climate_id, 'current_temperature') | float %}
        {{ temp|int if temp == temp|int else temp }} degrees
      {% else %}
        No climate entity set.
      {% endif %}

  # todo and shopping list
  - id: add_todo
    action:
      action: todo.add_item
      target:
        entity_id: |
          {{ states.todo
              | rejectattr('entity_id', 'eq', 'todo.shopping_list')
              | map(attribute='entity_id')
              | first }}
      data:
        item: "{{ args.item }}"
    response: >-
      Adding {{ args.item }} to todo list.

  - id: complete_todo
    action:
      action: todo.update_item
      target:
        entity_id: |
          {{ states.todo
              | rejectattr('entity_id', 'eq', 'todo.shopping_list')
              | map(attribute='entity_id')
              | first }}
      data:
        item: "{{ args.item }}"
        status: completed
    response: >-
      Completed {{ args.item }} on todo list.

  - id: add_to_shopping_list
    action:
      action: shopping_list.add_item
      data:
        name: "{{ args.item }}"
    response: >-
      Adding {{ args.item }} to shopping list.

  # media
  - id: control_media
    intent: |
      {% if args.action == 'pause' %}
      HassMediaPause
      {% elif args.action == 'resume' %}
      HassMediaUnpause
      {% elif args.action == 'next' %}
      HassMediaNext
      {% endif %}
    response: |
      {% if args.action == 'pause' %}
      Pausing media.
      {% elif args.action == 'resume' %}
      Resuming media.
      {% elif args.action == 'next' %}
      Skipping to next track.
      {% endif %}

  - id: media_volume
    intent:
      name: |
        {% if args.action == 'up' %}
        HassSetVolumeRelative
        {% elif args.action == 'down' %}
        HassSetVolumeRelative
        {% elif args.action == 'set' %}
        HassSetVolume
        {% endif %}
      slots:
        volume_step: |
          {% if args.action == 'up' %}
          up
          {% elif args.action == 'down' %}
          down
          {% else %}
          none
          {% endif %}
        volume_level: "{{ args.get('level') }}"
    response: |
      {% if args.action == 'increase' %}
      Increasing volume.
      {% elif args.action == 'decrease' %}
      Decreasing volume.
      {% elif args.action == 'set' %}
      Setting volume to {{ args.level }} percent.
      {% endif %}

  - id: play_music
    intent:
      name: HassMediaSearchAndPlay
      slots:
        search_query: "{{ args.query }}"
        media_class: "{{ args.get('query_type') }}"
    response: |
      Playing {{ args.query }}.

# -----------------------------------------------------------------------------

tools:
  - type: function
    function:
      name: nevermind
      description: Cancels or ignores the command.

  # date and time
  - type: function
    function:
      name: current_time
      description: Get the current time.
  - type: function
    function:
      name: current_date
      description: Get the current date.

  # timers
  - type: function
    function:
      name: set_timer
      description: Set a timer with a duration and optional name
      parameters:
        type: object
        properties:
          total_seconds:
            type: integer
            minimum: 0
          name:
            type: string
        anyOf:
          - required:
              - total_seconds
          - required:
              - total_seconds
              - name
  - type: function
    function:
      name: control_timer
      description: Pause, resume, or cancel an active timer.
      parameters:
        type: object
        properties:
          action:
            type: string
            enum:
              - pause
              - resume
              - cancel
        required:
          - action

  # Can't access timers via API
  # - type: function
  #   function:
  #     name: timer_status
  #     description: Get the time remaining for an active timer.

  # lights
  - type: function
    function:
      name: control lights
      description: Turn the lights in the current area on or off.
      parameters:
        type: object
        properties:
          action:
            type: string
            enum:
              - "on"
              - "off"
        required:
          - action
  - type: function
    function:
      name: set_brightness
      description: Set the brightness of the lights in the current area.
      parameters:
        type: object
        properties:
          brightness:
            type: integer
            minimum: 0
            maximum: 100
        required:
          - brightness

  # media
  - type: function
    function:
      name: control_media
      description: Pause, resume, or skip track on media player.
      parameters:
        type: object
        properties:
          action:
            type: string
            enum:
              - pause
              - resume
              - next
        required:
          - action
  - type: function
    function:
      name: media_volume
      description: Turn the volume of a media player up, down, or set it by percentage.
      parameters:
        type: object
        properties:
          action:
            type: string
            enum:
              - up
              - down
              - set
          target:
            type: string
            description: Device or area name
          level:
            type: integer
            minimum: 0
            maximum: 100
        anyOf:
          - required:
              - action
          - required:
              - action
              - level
  - type: function
    function:
      name: play_music
      description: Play music with a search query.
      parameters:
        type: object
        properties:
          query:
            type: string
          query_type:
            type: string
            enum:
              - artist
              - album
              - track
        anyOf:
          - required:
              - query
          - required:
              - query
              - query_type

  # weather and temperature
  - type: function
    function:
      name: weather_forecast
      description: Get the current weather forecast.
  - type: function
    function:
      name: get_temperature
      description: Get the current temperature of the thermostat.

  # todo and shopping list
  - type: function
    function:
      name: add_todo
      description: Add a task to the todo list.
      parameters:
        type: object
        properties:
          item:
            type: string
        anyOf:
          - required:
              - item
  - type: function
    function:
      name: complete_todo
      description: Mark a task as completed on the todo list.
      parameters:
        type: object
        properties:
          item:
            type: string
        anyOf:
          - required:
              - item
  - type: function
    function:
      name: add_to_shopping_list
      description: Add an item to the shopping list.
      parameters:
        type: object
        properties:
          item:
            type: string
        anyOf:
          - required:
              - item
```

<!-- Links -->
[conversation]: https://www.home-assistant.io/integrations/conversation/
[gemma4]: https://deepmind.google/models/gemma/gemma-4/
[llama.cpp]: https://github.com/ggml-org/llama.cpp
[actions]: https://www.home-assistant.io/docs/scripts/perform-actions/
[intents]: https://developers.home-assistant.io/docs/intent_builtin/
[tools]: https://developers.openai.com/api/docs/guides/function-calling
[templates]: https://www.home-assistant.io/docs/templating/
[llama-cpp-python]: https://llama-cpp-python.readthedocs.io/en/latest/
