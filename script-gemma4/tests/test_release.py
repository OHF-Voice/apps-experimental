import unittest
from pathlib import Path

import yaml

from const import APP_VERSION


class ReleaseMetadataTests(unittest.TestCase):
    def test_config_version_matches_runtime_version(self):
        project_dir = Path(__file__).parent.parent
        config = yaml.safe_load((project_dir / "config.yaml").read_text("utf-8"))

        self.assertEqual(APP_VERSION, config["version"])

    def test_flash_attention_is_enabled_by_default(self):
        project_dir = Path(__file__).parent.parent
        config = yaml.safe_load((project_dir / "config.yaml").read_text("utf-8"))

        self.assertIs(True, config["options"]["flash_attention"])


if __name__ == "__main__":
    unittest.main()
