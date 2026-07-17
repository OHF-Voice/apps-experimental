"""Wrapper for Home Assistant REST/Websocket API."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse, urlunparse

import aiohttp

from models import ATTR_FRIENDLY_NAME, Area, Entity, Floor

_LOGGER = logging.getLogger(__name__)


class HomeAssistantError(Exception):
    pass


@dataclass
class HomeInfo:
    """Snapshot of Home Assistant used to generate the tool set.

    Only conversation-exposed entities are included (those the model is allowed
    to act on), but *all* areas and floors are included so area/floor slots can
    target anywhere. Names include aliases.
    """

    entities: Dict[str, Entity] = field(default_factory=dict)
    areas: Dict[str, Area] = field(default_factory=dict)
    floors: Dict[str, Floor] = field(default_factory=dict)

    @property
    def available_domains(self) -> Set[str]:
        """Entity domains that have at least one exposed entity."""
        return {entity.domain for entity in self.entities.values()}

    def entity_names(self, domains: Optional[Set[str]] = None) -> list:
        """Sorted, de-duplicated entity names (+ aliases), optionally by domain."""
        names: Set[str] = set()
        for entity in self.entities.values():
            if domains is not None and entity.domain not in domains:
                continue
            for name in entity.names:
                name = name.strip()
                if name:
                    names.add(name)
        return sorted(names)

    def device_classes(self, domains: Optional[Set[str]] = None) -> list:
        """Sorted device classes present on exposed entities, optionally by domain."""
        classes: Set[str] = set()
        for entity in self.entities.values():
            if domains is not None and entity.domain not in domains:
                continue
            device_class = entity.device_class
            if device_class:
                classes.add(device_class)
        return sorted(classes)

    def area_names(self) -> list:
        names: Set[str] = set()
        for area in self.areas.values():
            for name in area.names:
                name = name.strip()
                if name:
                    names.add(name)
        return sorted(names)

    def floor_names(self) -> list:
        names: Set[str] = set()
        for floor in self.floors.values():
            for name in floor.names:
                name = name.strip()
                if name:
                    names.add(name)
        return sorted(names)


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

        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = f"{parsed.path}/websocket"
        self.websocket_api_url = urlunparse(
            parsed._replace(
                scheme=scheme, path=path, params="", query="", fragment=""
            )
        )

    async def get_home_info(self) -> HomeInfo:
        """Fetch exposed entities (with aliases) plus all areas and floors."""
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

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
                    {"type": "auth", "access_token": self.token}
                )
                msg = await websocket.receive_json()
                assert msg["type"] == "auth_ok", msg

                # Exposed (to conversation) entities
                await websocket.send_json(
                    {"id": next_id(), "type": "homeassistant/expose_entity/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                exposed_entity_ids = {
                    entity_id
                    for entity_id, exposed_info in msg["result"][
                        "exposed_entities"
                    ].items()
                    if exposed_info.get("conversation")
                }

                # States (for attributes / friendly names)
                await websocket.send_json({"id": next_id(), "type": "get_states"})
                msg = await websocket.receive_json()
                assert msg["success"], msg
                state_attrs: Dict[str, Dict[str, Any]] = {}
                for state_data in msg["result"]:
                    entity_id = state_data["entity_id"]
                    if entity_id in exposed_entity_ids:
                        state_attrs[entity_id] = state_data.get("attributes", {})

                # Floors (all)
                await websocket.send_json(
                    {"id": next_id(), "type": "config/floor_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                for floor_data in msg["result"]:
                    floor_id = floor_data["floor_id"]
                    floors[floor_id] = Floor(
                        floor_id=floor_id,
                        name=floor_data["name"],
                        aliases=floor_data.get("aliases"),
                    )

                # Areas (all)
                await websocket.send_json(
                    {"id": next_id(), "type": "config/area_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                for area_data in msg["result"]:
                    area_id = area_data["area_id"]
                    areas[area_id] = Area(
                        area_id=area_id,
                        name=area_data["name"],
                        aliases=area_data.get("aliases"),
                        floor_id=area_data.get("floor_id"),
                    )

                # Devices (for entity area fallback)
                await websocket.send_json(
                    {"id": next_id(), "type": "config/device_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                devices = {d["id"]: d for d in msg["result"]}

                # Entity registry entries for exposed entities (names, aliases, area)
                if exposed_entity_ids:
                    await websocket.send_json(
                        {
                            "id": next_id(),
                            "type": "config/entity_registry/get_entries",
                            "entity_ids": list(exposed_entity_ids),
                        }
                    )
                    msg = await websocket.receive_json()
                    assert msg["success"], msg
                    entries = msg["result"]
                else:
                    entries = {}

                for entity_id in exposed_entity_ids:
                    entity_info = entries.get(entity_id) or {}
                    if entity_info.get("disabled_by") is not None:
                        # Skip disabled entities
                        continue

                    attributes = state_attrs.get(entity_id, {})

                    name = (
                        entity_info.get("name")
                        or entity_info.get("original_name")
                        or attributes.get(ATTR_FRIENDLY_NAME)
                        or ""
                    )

                    aliases = list(filter(None, entity_info.get("aliases") or []))

                    if not name.strip() and not aliases:
                        # Exposed but unnameable (e.g. an orphaned entity with
                        # no registry name and no state): it cannot be targeted
                        # by voice, would blank out the web UI listing, and
                        # would wrongly keep its domain "available" for tool
                        # filtering.
                        _LOGGER.warning(
                            "Skipping exposed entity with no name or aliases: %s",
                            entity_id,
                        )
                        continue

                    # Area: entity area, falling back to its device's area
                    area_id = entity_info.get("area_id")
                    if not area_id:
                        device_id = entity_info.get("device_id")
                        if device_id:
                            area_id = (devices.get(device_id) or {}).get("area_id")

                    entities[entity_id] = Entity(
                        entity_id=entity_id,
                        name=name,
                        aliases=aliases or None,
                        attributes=attributes,
                        area_id=area_id,
                    )

        home_info = HomeInfo(entities=entities, areas=areas, floors=floors)
        _LOGGER.debug(
            "Loaded %s exposed entit(ies) across domains %s, %s area(s), %s floor(s)",
            len(entities),
            sorted(home_info.available_domains),
            len(areas),
            len(floors),
        )
        return home_info

    async def handle_intent(
        self,
        intent_name: str,
        language: str,
        *,
        data: Optional[Mapping[str, Any]] = None,
        device_id: Optional[str] = None,
        satellite_id: Optional[str] = None,
        assistant: Optional[str] = "conversation",
    ) -> Dict[str, Any]:
        """Handle intent with REST API and return response."""
        headers = {"Authorization": f"Bearer {self.token}"}

        async with aiohttp.ClientSession() as session:
            web_response = await session.post(
                f"{self.api_url}/intent/handle",
                json={
                    "name": intent_name,
                    "data": data,
                    "language": language,
                    "assistant": assistant,
                    "device_id": device_id,
                    "satellite_id": satellite_id,
                },
                headers=headers,
            )
            assert web_response.status == 200
            return await web_response.json()
