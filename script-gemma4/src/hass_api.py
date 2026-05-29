"""Wrapper for Home Assistant REST/Websocket API."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse, urlunparse

import aiohttp

from models import ATTR_FRIENDLY_NAME, Area, Entity, Floor, State

_LOGGER = logging.getLogger(__name__)

SEARCH_MEDIA = 4194304  # MediaPlayerEntityFeature


class HomeAssistantError(Exception):
    pass


@dataclass
class HomeAssistantInfo:
    states: Dict[str, State]
    entities: Dict[str, Entity]
    areas: Dict[str, Area]
    floors: Dict[str, Floor]


@dataclass
class SatelliteInfo:
    entity_id: Optional[str] = None
    device_id: Optional[str] = None
    area_id: Optional[str] = None
    floor_id: Optional[str] = None
    media_player_id: Optional[str] = None
    music_player_id: Optional[str] = None
    music_assistant_id: Optional[str] = None

    def as_dict(self) -> Dict[str, str]:
        info_dict: Dict[str, str] = {}
        if self.entity_id:
            info_dict["entity_id"] = self.entity_id
        if self.device_id:
            info_dict["device_id"] = self.device_id
        if self.area_id:
            info_dict["area_id"] = self.area_id
        if self.floor_id:
            info_dict["floor_id"] = self.floor_id
        if self.media_player_id:
            info_dict["media_player_id"] = self.media_player_id
        if self.music_player_id:
            info_dict["music_player_id"] = self.music_player_id
        if self.music_assistant_id:
            info_dict["music_assistant_id"] = self.music_assistant_id

        return info_dict


@dataclass
class Tool:
    name: str
    tool: Dict[str, Any]
    name_map: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: defaultdict(dict)
    )


class HomeAssistant:
    """API to Home Assistant."""

    def __init__(
        self,
        token: str,
        api_url: str = "http://homeassistant.local:8123/api",
    ) -> None:
        self.token = token
        self.api_url = api_url.rstrip("/")

        # Get websocket API URL
        parsed = urlparse(self.api_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        # Convert scheme
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = f"{parsed.path}/websocket"
        self.websocket_api_url = urlunparse(
            parsed._replace(
                scheme=scheme,
                path=path,
                params="",
                query="",
                fragment="",
            )
        )

    async def get_satellite_info(
        self, device_id: Optional[str] = None, satellite_id: Optional[str] = None
    ) -> SatelliteInfo:
        satellite_info = SatelliteInfo(device_id=device_id, entity_id=satellite_id)

        if (satellite_info.device_id is None) and (satellite_info.entity_id is None):
            # Can't get any more info
            return satellite_info

        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self.websocket_api_url, max_msg_size=0
            ) as websocket:
                # Authenticate
                msg = await websocket.receive_json()
                assert msg["type"] == "auth_required", msg

                await websocket.send_json(
                    {
                        "type": "auth",
                        "access_token": self.token,
                    },
                )

                msg = await websocket.receive_json()
                assert msg["type"] == "auth_ok", msg

                # Devices
                await websocket.send_json(
                    {"id": next_id(), "type": "config/device_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                devices = {
                    device_info["id"]: device_info for device_info in msg["result"]
                }

                # Areas
                await websocket.send_json(
                    {"id": next_id(), "type": "config/area_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                areas = {area_info["area_id"]: area_info for area_info in msg["result"]}

                # Floors
                # await websocket.send_json(
                #     {"id": next_id(), "type": "config/floor_registry/list"}
                # )
                # msg = await websocket.receive_json()
                # assert msg["success"], msg
                # floors = {
                #     floor_info["floor_id"]: floor_info for floor_info in msg["result"]
                # }

                # States
                # await websocket.send_json({"id": next_id(), "type": "get_states"})
                # msg = await websocket.receive_json()
                # assert msg["success"], msg
                # states = {state["entity_id"]: state for state in msg["result"]}

                # Media players
                await websocket.send_json(
                    {"id": next_id(), "type": "config/entity_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                media_players = {
                    mp_info["entity_id"]: mp_info
                    for mp_info in msg["result"]
                    if mp_info["entity_id"].startswith("media_player.")
                    and (mp_info.get("disabled_by") is None)
                }

                # Add area/floor info to media players
                for mp_id, mp_info in media_players.items():
                    mp_area_id = mp_info.get("area_id")
                    mp_floor_id: Optional[str] = None
                    if not mp_area_id:
                        mp_device_id = mp_info.get("device_id")
                        if mp_device_id:
                            mp_area_id = devices.get(mp_device_id, {}).get("area_id")

                    if mp_area_id:
                        mp_info["area_id"] = mp_area_id
                        mp_floor_id = areas.get(mp_area_id, {}).get("floor_id")
                        if mp_floor_id:
                            mp_info["floor_id"] = mp_floor_id

                # Music players (that support SEARCH_MEDIA for Music Assistant)
                music_players: Dict[str, Dict[str, Any]] = {}
                for mp_id, mp_info in media_players.items():
                    if mp_info.get("platform") != "music_assistant":
                        continue

                    if not mp_info.get("config_entry_id"):
                        continue

                    music_players[mp_id] = mp_info

                if satellite_info.entity_id:
                    # Get area of assist_satellite entity
                    await websocket.send_json(
                        {
                            "id": next_id(),
                            "type": "config/entity_registry/get_entries",
                            "entity_ids": [satellite_info.entity_id],
                        }
                    )
                    msg = await websocket.receive_json()
                    assert msg["success"], msg
                    satellite_dict = next(iter(msg["result"].values()))
                    satellite_area_id = satellite_dict.get("area_id")
                    if satellite_area_id:
                        return satellite_area_id

                    # Use device area
                    satellite_info.device_id = satellite_dict.get(
                        "device_id", device_id
                    )
                    if satellite_info.device_id:
                        satellite_info.area_id = devices.get(device_id, {}).get(
                            "area_id"
                        )

                if satellite_info.device_id:
                    # Get area from device instead
                    satellite_info.area_id = devices.get(device_id, {}).get("area_id")

                    # Look for media/music player on the same device
                    for mp_id, mp_info in media_players.items():
                        if mp_info.get("device_id") != satellite_info.device_id:
                            continue

                        if not satellite_info.media_player_id:
                            satellite_info.media_player_id = mp_id
                            _LOGGER.debug("Selected media player by device: %s", mp_id)

                        if (not satellite_info.music_player_id) and (
                            mp_id in music_players
                        ):
                            satellite_info.music_player_id = mp_id
                            satellite_info.music_assistant_id = mp_info.get(
                                "config_entry_id"
                            )
                            _LOGGER.debug("Selected music player by device: %s", mp_id)

                if satellite_info.area_id:
                    satellite_info.floor_id = areas.get(satellite_info.area_id, {}).get(
                        "floor_id"
                    )

                # Look for media/music player in the same area
                if (
                    (not satellite_info.media_player_id)
                    or (not satellite_info.music_player_id)
                ) and satellite_info.area_id:
                    for mp_id, mp_info in media_players.items():
                        if mp_info.get("area_id") != satellite_info.area_id:
                            continue

                        if not satellite_info.media_player_id:
                            satellite_info.media_player_id = mp_id
                            _LOGGER.debug("Selected media player by area: %s", mp_id)

                        if (not satellite_info.music_player_id) and (
                            mp_id in music_players
                        ):
                            satellite_info.music_player_id = mp_id
                            satellite_info.music_assistant_id = mp_info.get(
                                "config_entry_id"
                            )
                            _LOGGER.debug("Selected music player by area: %s", mp_id)

                # Look for media/music player on the same floor
                if (
                    (not satellite_info.media_player_id)
                    or (not satellite_info.music_player_id)
                ) and satellite_info.floor_id:
                    for mp_id, mp_info in media_players.items():
                        if mp_info.get("floor_id") != satellite_info.floor_id:
                            continue

                        if not satellite_info.media_player_id:
                            satellite_info.media_player_id = mp_id
                            _LOGGER.debug("Selected media player by floor: %s", mp_id)

                        if (not satellite_info.music_player_id) and (
                            mp_id in music_players
                        ):
                            satellite_info.music_player_id = mp_id
                            satellite_info.music_assistant_id = mp_info.get(
                                "config_entry_id"
                            )
                            _LOGGER.debug("Selected music player by floor: %s", mp_id)

        return satellite_info

    async def get_script_tools(self, info: HomeAssistantInfo) -> List[Tool]:
        tools: List[Tool] = []
        now = datetime.now()

        # Area/floor names
        area_name_map = {}
        area_names = set()
        for area in info.areas.values():
            for area_name in area.names:
                if area_name:
                    area_names.add(area_name)
                    area_name_map[area_name] = area.area_id
        area_names_sorted = sorted(area_names)

        floor_name_map = {}
        floor_names = set()
        for floor in info.floors.values():
            for floor_name in floor.names:
                if floor_name:
                    floor_names.add(floor_name)
                    floor_name_map[floor_name] = floor.floor_id
        floor_names_sorted = sorted(floor_names)

        # ---
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self.websocket_api_url, max_msg_size=0
            ) as websocket:
                # Authenticate
                msg = await websocket.receive_json()
                assert msg["type"] == "auth_required", msg

                await websocket.send_json(
                    {
                        "type": "auth",
                        "access_token": self.token,
                    },
                )

                msg = await websocket.receive_json()
                assert msg["type"] == "auth_ok", msg

                # Get exposed entities
                await websocket.send_json(
                    {"id": next_id(), "type": "homeassistant/expose_entity/list"}
                )

                msg = await websocket.receive_json()
                assert msg["success"], msg

                exposed_scripts = set()
                for entity_id, exposed_info in msg["result"][
                    "exposed_entities"
                ].items():
                    domain, script_id = entity_id.split(".", maxsplit=1)
                    if domain != "script":
                        continue

                    if not exposed_info.get("conversation"):
                        continue

                    exposed_scripts.add(script_id)

                await websocket.send_json(
                    {
                        "id": next_id(),
                        "type": "get_services",
                    }
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg

                scripts = msg["result"]["script"]
                for script_id in sorted(exposed_scripts):
                    script = scripts[script_id]
                    tool_dict: Dict[str, Any] = {
                        "type": "function",
                        "function": {
                            "name": script_id,
                            "description": script.get("description") or script["name"],
                        },
                    }
                    tool = Tool(name=script_id, tool=tool_dict)
                    tools.append(tool)

                    fields = script.get("fields")
                    if not fields:
                        continue

                    # Convert fields to tools
                    props: Dict[str, Any] = {}
                    params = tool_dict["function"].setdefault(
                        "parameters", {"type": "object", "properties": props}
                    )
                    required_fields = set()
                    for field_key, field_info in fields.items():
                        if field_info.get("required"):
                            required_fields.add(field_key)

                        field_description = field_info.get("description") or ""

                        # Default to string type
                        field_prop: Dict[str, Any] = {"type": "string"}
                        props[field_key] = field_prop

                        selector = field_info.get("selector")
                        if selector:
                            if "text" in selector:
                                pass  # already a string
                            elif "number" in selector:
                                field_prop["type"] = "number"
                                num_selector = selector["number"]
                                if "min" in num_selector:
                                    field_prop["minimum"] = num_selector["min"]
                                if "max" in num_selector:
                                    field_prop["maximum"] = num_selector["max"]
                                if "step" in num_selector:
                                    step = num_selector["step"]
                                    if step == 1:
                                        field_prop["type"] = "integer"
                                    else:
                                        field_prop["multipleOf"] = step
                            elif "boolean" in selector:
                                field_prop["type"] = "boolean"
                            elif "select" in selector:
                                field_prop["enum"] = selector["select"]["options"]
                            elif "date" in selector:
                                field_prop["format"] = "date"
                                field_description += f"\nDate in YYYY-MM-DD format. The current year is {now.year}"
                            elif "time" in selector:
                                field_prop["format"] = "time"
                                field_description += "\nTime in HH:MM:SS format, or HH:MM if seconds are not needed"
                            elif "datetime" in selector:
                                field_prop["format"] = "date-time"
                                field_description += f"\nISO 8601 datetime. Include timezone offset when known. The current year is {now.year}"
                            elif "duration" in selector:
                                field_prop["format"] = "duration"
                                field_description += "\nDuration in HH:MM:SS format, or HH:MM if seconds are not needed"
                            elif "color_rgb" in selector:
                                field_prop["type"] = "array"
                                field_prop["items"] = {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 255,
                                }
                                field_prop["minItems"] = 3
                                field_prop["maxItems"] = 3
                                field_description += (
                                    "\nRGB color as [red, green, blue], each 0-255."
                                )
                            elif "color_temp" in selector:
                                field_prop["type"] = "integer"
                                field_prop["minimum"] = 2000
                                field_prop["maximum"] = 6500
                                field_description += "\nColor temperature in kelvin, e.g. 2700 for warm white or 6500 for cool daylight."
                            elif ("area" in selector) and area_names_sorted:
                                field_prop["enum"] = area_names_sorted
                                tool.name_map[field_key] = area_name_map
                            elif ("floor" in selector) and floor_names_sorted:
                                field_prop["enum"] = floor_names_sorted
                                tool.name_map[field_key] = floor_name_map
                            elif ("entity" in selector) and info.entities:
                                entity_names = set()
                                entity_filters = selector["entity"].get("filter", [])
                                filter_domains: Optional[Set[str]] = None
                                if entity_filters:
                                    filter_domains = entity_filters[0].get("domain")
                                    if isinstance(filter_domains, str):
                                        filter_domains = {filter_domains}
                                    elif filter_domains is not None:
                                        filter_domains = set(filter_domains)

                                entity_name_map = {}
                                for entity in info.entities.values():
                                    if filter_domains and (
                                        entity.domain not in filter_domains
                                    ):
                                        continue

                                    for entity_name in entity.names:
                                        if entity_name:
                                            entity_names.add(entity_name)
                                            entity_name_map[entity_name] = (
                                                entity.entity_id
                                            )

                                entity_names_sorted = sorted(entity_names)
                                field_prop["enum"] = entity_names_sorted
                                tool.name_map[field_key] = entity_name_map

                        if field_description:
                            field_prop["description"] = field_description.strip()

                    if required_fields:
                        params["required"] = sorted(required_fields)

        return tools

    async def get_home_info(self) -> HomeAssistantInfo:
        """Get necessary information for intent recognition."""
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        states: Dict[str, State] = {}
        entities: Dict[str, Entity] = {}
        areas: Dict[str, Area] = {}
        floors: Dict[str, Floor] = {}

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self.websocket_api_url, max_msg_size=0
            ) as websocket:
                # Authenticate
                msg = await websocket.receive_json()
                assert msg["type"] == "auth_required", msg

                await websocket.send_json(
                    {
                        "type": "auth",
                        "access_token": self.token,
                    },
                )

                msg = await websocket.receive_json()
                assert msg["type"] == "auth_ok", msg

                # Get exposed entities
                await websocket.send_json(
                    {"id": next_id(), "type": "homeassistant/expose_entity/list"}
                )

                msg = await websocket.receive_json()
                assert msg["success"], msg

                exposed_entity_ids = set()
                for entity_id, exposed_info in msg["result"][
                    "exposed_entities"
                ].items():
                    if exposed_info.get("conversation"):
                        exposed_entity_ids.add(entity_id)

                await websocket.send_json(
                    {
                        "id": next_id(),
                        "type": "get_states",
                    }
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                for state_data in msg["result"]:
                    entity_id = state_data["entity_id"]
                    if entity_id not in exposed_entity_ids:
                        continue

                    states[entity_id] = State(
                        entity_id=entity_id,
                        state=state_data["state"],
                        attributes=state_data.get("attributes", {}),
                    )

                # Floors
                await websocket.send_json(
                    {"id": next_id(), "type": "config/floor_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                for floor_data in msg["result"]:
                    floor_id = floor_data["floor_id"]
                    floors[floor_id] = Floor(
                        floor_id=floor_id,
                        name=floor_data["name"].strip(),
                        aliases=floor_data.get("aliases"),
                    )

                # Areas
                await websocket.send_json(
                    {"id": next_id(), "type": "config/area_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                for area_data in msg["result"]:
                    area_id = area_data["area_id"]
                    areas[area_id] = Area(
                        area_id=area_id,
                        name=area_data["name"].strip(),
                        aliases=area_data.get("aliases"),
                        floor_id=area_data.get("floor_id"),
                    )

                # Devices
                await websocket.send_json(
                    {"id": next_id(), "type": "config/device_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                devices = {
                    device_info["id"]: device_info for device_info in msg["result"]
                }

                # Contains aliases
                # Check area_id as well as area of device_id
                # Use original_device_class
                await websocket.send_json(
                    {
                        "id": next_id(),
                        "type": "config/entity_registry/get_entries",
                        "entity_ids": list(exposed_entity_ids),
                    }
                )

                msg = await websocket.receive_json()
                assert msg["success"], msg
                for entity_id, entity_info in msg["result"].items():
                    name = None
                    names: List[str] = []

                    if entity_info:
                        if entity_info.get("disabled_by") is not None:
                            # Skip disabled entities
                            continue

                        name = (
                            entity_info.get("name", "") or entity_info["original_name"]
                        )
                        if entity_info.get("aliases"):
                            names.extend(filter(None, entity_info["aliases"]))

                    entity_area_id = None
                    if entity_info:
                        entity_area_id = entity_info.get("area_id")

                        if not entity_area_id:
                            # Try to get area from device
                            entity_device_id = entity_info.get("device_id")
                            if entity_device_id:
                                device_info = devices.get(entity_device_id)
                                if device_info:
                                    entity_area_id = device_info.get("area_id")

                    attributes: Dict[str, Any] = {}
                    state_data = states.get(entity_id)
                    if state_data:
                        attributes = state_data.attributes

                    if not name:
                        # Try friendly name
                        name = attributes.get(ATTR_FRIENDLY_NAME, "")

                    if name:
                        name = name.strip()
                        if state_data:
                            state_data.entity_name = name

                    entities[entity_id] = Entity(
                        entity_id=entity_id,
                        name=name,
                        aliases=names if names else None,
                        attributes=attributes,
                        area_id=entity_area_id,
                    )

        _LOGGER.debug(
            "Loaded %s entities, %s area(s), %s floor(s)",
            len(entities),
            len(areas),
            len(floors),
        )

        return HomeAssistantInfo(
            states=states, entities=entities, areas=areas, floors=floors
        )

    async def call_service(
        self,
        domain: str,
        service: str,
        service_data: Optional[Dict[str, Any]] = None,
        target: Optional[Dict[str, Any]] = None,
        return_response: bool = False,
    ) -> Optional[Dict[str, Any]]:
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self.websocket_api_url, max_msg_size=0
            ) as websocket:
                # Authenticate
                msg = await websocket.receive_json()
                assert msg["type"] == "auth_required", msg

                await websocket.send_json(
                    {
                        "type": "auth",
                        "access_token": self.token,
                    },
                )

                msg = await websocket.receive_json()
                assert msg["type"] == "auth_ok", msg

                _LOGGER.debug(
                    "Calling service %s.%s with target=%s, data=%s",
                    domain,
                    service,
                    target,
                    service_data,
                )

                await websocket.send_json(
                    {
                        "id": next_id(),
                        "type": "call_service",
                        "domain": domain,
                        "service": service,
                        "service_data": service_data or {},
                        "target": target or {},
                        "return_response": return_response,
                    },
                )
                msg = await websocket.receive_json()
                if not msg["success"]:
                    raise HomeAssistantError(msg["error"]["message"])

                return msg.get("result", {}).get("response")
