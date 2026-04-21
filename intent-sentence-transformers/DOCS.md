# Sentence Transformers Agent

Uses [sentence transformers][sentence_transformers] to match voice commands to [intents][] or [actions][].

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
    
## Future Ideas

- Different response for when intent or action fails
- Better distance metric than just cosine
    
<!-- Links -->
[sentence_transformers]: https://huggingface.co/sentence-transformers
[actions]: https://www.home-assistant.io/docs/scripts/perform-actions/
[intents]: https://developers.home-assistant.io/docs/intent_builtin/

[assist_satellite]: https://www.home-assistant.io/integrations/assist_satellite/
