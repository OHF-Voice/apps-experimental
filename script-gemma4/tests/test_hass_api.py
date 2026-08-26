import unittest

from hass_api import SatelliteInfo, _apply_satellite_registry_info


class SatelliteRegistryTests(unittest.TestCase):
    def test_direct_entity_area_keeps_satellite_object(self):
        satellite = SatelliteInfo(entity_id="assist_satellite.kitchen")

        _apply_satellite_registry_info(
            satellite,
            {"area_id": "kitchen", "device_id": "device-1"},
            {"device-1": {"area_id": "office"}},
        )

        self.assertIsInstance(satellite, SatelliteInfo)
        self.assertEqual("kitchen", satellite.area_id)
        self.assertEqual("device-1", satellite.device_id)

    def test_derived_device_id_is_used_for_area(self):
        satellite = SatelliteInfo(entity_id="assist_satellite.office")

        _apply_satellite_registry_info(
            satellite,
            {"area_id": None, "device_id": "device-2"},
            {"device-2": {"area_id": "office"}},
        )

        self.assertEqual("device-2", satellite.device_id)
        self.assertEqual("office", satellite.area_id)

    def test_input_device_survives_empty_registry_device(self):
        satellite = SatelliteInfo(device_id="device-3")

        _apply_satellite_registry_info(
            satellite,
            {"area_id": None, "device_id": None},
            {"device-3": {"area_id": "bedroom"}},
        )

        self.assertEqual("device-3", satellite.device_id)
        self.assertEqual("bedroom", satellite.area_id)


if __name__ == "__main__":
    unittest.main()
