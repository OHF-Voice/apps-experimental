from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import aiohttp


@dataclass
class InfoForRecognition:
    """Information gathered from Home Assistant for intent recognition."""

    current_area_id: Optional[str]
    current_floor_id: Optional[str]
    satellite_devices: Dict[str, str]


class HomeAssistant:
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

    async def get_info(
        self, device_id: Optional[str] = None, satellite_id: Optional[str] = None
    ) -> InfoForRecognition:
        """Get necessary information for intent recognition."""
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        current_area_id: Optional[str] = None
        current_floor_id: Optional[str] = None

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

                # Areas
                await websocket.send_json(
                    {"id": next_id(), "type": "config/area_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg

                # Devices
                await websocket.send_json(
                    {"id": next_id(), "type": "config/device_registry/list"}
                )
                msg = await websocket.receive_json()
                assert msg["success"], msg
                devices = {
                    device_info["id"]: device_info for device_info in msg["result"]
                }

                satellite_ids = set()
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
                    domain = entity_id.split(".", maxsplit=1)[0]
                    if domain == "assist_satellite":
                        satellite_ids.add(entity_id)

                # Get preferred area
                if satellite_id:
                    # Get area of assist_satellite entity
                    await websocket.send_json(
                        {
                            "id": next_id(),
                            "type": "config/entity_registry/get_entries",
                            "entity_ids": [satellite_id],
                        }
                    )
                    msg = await websocket.receive_json()
                    assert msg["success"], msg
                    satellite_info = next(iter(msg["result"].values()))
                    satellite_area_id = satellite_info.get("area_id")
                    if satellite_area_id:
                        current_area_id = satellite_area_id
                    else:
                        # Use device area
                        satellite_device_id = satellite_info.get("device_id")
                        if satellite_device_id:
                            current_area_id = devices.get(satellite_device_id, {}).get(
                                "area_id"
                            )
                elif device_id:
                    # Get area from device instead
                    current_area_id = devices.get(device_id, {}).get("area_id")

                # Get satellite devices
                satellite_devices: Dict[str, str] = {}
                if satellite_ids:
                    await websocket.send_json(
                        {
                            "id": next_id(),
                            "type": "config/entity_registry/get_entries",
                            "entity_ids": list(satellite_ids),
                        }
                    )

                    msg = await websocket.receive_json()
                    assert msg["success"], msg
                    for entity_id, entity_info in msg["result"].items():
                        device_id = entity_info.get("device_id")
                        if device_id:
                            satellite_devices[entity_id] = device_id

        return InfoForRecognition(
            current_area_id=current_area_id,
            current_floor_id=current_floor_id,
            satellite_devices=satellite_devices,
        )

    async def render_template(
        self, template: str, variables: Optional[Dict[str, Any]] = None
    ) -> Any:
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

                await websocket.send_json(
                    {
                        "id": next_id(),
                        "type": "render_template",
                        "template": template,
                        "variables": variables or {},
                    },
                )
                msg = await websocket.receive_json()
                assert msg["type"] == "result"
                assert msg["success"], msg

                msg = await websocket.receive_json()
                assert msg["type"] == "event"

                return msg["event"]["result"]
