# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LicenseFilesTests(unittest.TestCase):
    def test_code_license_is_apache_2(self) -> None:
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0", text)

    def test_documentation_license_is_cc_by_4(self) -> None:
        text = (ROOT / "docs" / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Attribution 4.0 International", text)

    def test_license_scope_is_documented(self) -> None:
        text = (ROOT / "LICENSING.md").read_text(encoding="utf-8")
        self.assertIn("Apache-2.0", text)
        self.assertIn("CC BY 4.0", text)
        self.assertIn("AGPLv3", text)

    def test_third_party_notices_cover_rwpp(self) -> None:
        text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn("RWPP", text)
        self.assertIn("AGPL-3.0", text)

    def test_readme_points_to_license_documents(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for name in ["LICENSING.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "NOTICE"]:
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
