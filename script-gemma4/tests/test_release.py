import unittest
from pathlib import Path

import yaml

from const import APP_VERSION
from gemma4_recognizer import DEFAULT_MAX_TOKENS


class ReleaseMetadataTests(unittest.TestCase):
    def test_config_version_matches_runtime_version(self):
        project_dir = Path(__file__).parent.parent
        config = yaml.safe_load((project_dir / "config.yaml").read_text("utf-8"))

        self.assertEqual(APP_VERSION, config["version"])

    def test_flash_attention_is_enabled_by_default(self):
        project_dir = Path(__file__).parent.parent
        config = yaml.safe_load((project_dir / "config.yaml").read_text("utf-8"))

        self.assertIs(True, config["options"]["flash_attention"])

    def test_generation_limit_matches_runtime_default(self):
        project_dir = Path(__file__).parent.parent
        config = yaml.safe_load((project_dir / "config.yaml").read_text("utf-8"))

        self.assertEqual(DEFAULT_MAX_TOKENS, config["options"]["max_tokens"])


if __name__ == "__main__":
    unittest.main()
