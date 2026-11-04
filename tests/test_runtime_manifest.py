# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeManifestTests(unittest.TestCase):
    def test_manifest_pins_verified_versions(self) -> None:
        manifest = json.loads((ROOT / "vendor" / "rwpp" / "pinned-versions.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["runtime_id"], "rwpp-1.15")
        self.assertEqual(manifest["game"]["version"], "1.15")
        self.assertEqual(manifest["game"]["game_code"], 176)
        self.assertEqual(manifest["game"]["build_number"], 28)
        self.assertEqual(manifest["game_lib"]["version"], "v3-1.15")
        self.assertEqual(manifest["java"]["major"], 25)
        self.assertRegex(manifest["rwpp"]["commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(manifest["rwml"]["commit"], r"^[0-9a-f]{40}$")

    def test_build_script_contains_required_compatibility_steps(self) -> None:
        script = (ROOT / "scripts" / "build_rwpp_compat.ps1").read_text(encoding="utf-8")
        self.assertIn('game-lib = "v3-1.15"', script)
        self.assertIn("LocaleEngine.translateMapName", script)
        self.assertIn(":desktop:publishToMavenLocal", script)
        self.assertIn(":rwpp-server:installDist", script)
        self.assertIn("check_environment.ps1", script)
        self.assertIn("smoke_rwpp.ps1", script)


if __name__ == "__main__":
    unittest.main()
