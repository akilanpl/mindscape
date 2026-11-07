"""Preservation protects research while allowing a maintained project overview."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/v3/preservation.py"
SPEC = importlib.util.spec_from_file_location("v3_preservation", SCRIPT)
preservation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preservation)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.previous_cwd = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, self.previous_cwd)
        self.git("init", "-q")
        self.git("config", "user.name", "Preservation test")
        self.git("config", "user.email", "preservation@example.test")
        Path("README.md").write_text("Historical overview\n")
        Path("research.json").write_text('{"result": 1}\n')
        self.git("add", ".")
        self.git("commit", "-qm", "Frozen research")
        self.base = self.git("rev-parse", "HEAD").strip()
        self.git("tag", "frozen-release")
        Path("configs/v3").mkdir(parents=True)
        Path("configs/v3/preserved_tags.json").write_text(
            json.dumps({"frozen-release": self.base})
        )
        self.base_patch = patch.object(preservation, "BASE", self.base)
        self.base_patch.start()
        self.addCleanup(self.base_patch.stop)

    def git(self, *args):
        return subprocess.check_output(["git", *args], text=True)

    def test_unchanged_baseline(self):
        receipt = preservation.verify()
        self.assertTrue(receipt["historical_files_unchanged"])
        self.assertTrue(receipt["frozen_research_files_unchanged"])
        self.assertEqual(receipt["documentation_updates"], {})

    def test_readme_update_is_recorded_and_original_is_recoverable(self):
        Path("README.md").write_text("Maintained overview\n")
        receipt = preservation.verify()
        self.assertFalse(receipt["historical_files_unchanged"])
        self.assertTrue(receipt["frozen_research_files_unchanged"])
        update = receipt["documentation_updates"]["README.md"]
        self.assertNotEqual(update["baseline_blob"], update["current_blob"])
        self.assertEqual(self.git("show", update["historical_source"]), "Historical overview\n")

    def test_research_edit_is_rejected_even_with_updated_readme(self):
        Path("README.md").write_text("Maintained overview\n")
        Path("research.json").write_text('{"result": 2}\n')
        with self.assertRaisesRegex(ValueError, "Historical file changed: research.json"):
            preservation.verify()

    def test_additional_tag_does_not_rewrite_frozen_tags(self):
        self.git("tag", "preservation-snapshot")
        receipt = preservation.verify()
        self.assertEqual(receipt["verified_release_tags"], {"frozen-release": self.base})
        self.assertIn("preservation-snapshot", receipt["tags"])

    def test_missing_release_tag_is_rejected(self):
        self.git("tag", "-d", "frozen-release")
        with self.assertRaisesRegex(ValueError, "Release tags changed"):
            preservation.verify()

    def test_retargeted_release_tag_is_rejected(self):
        Path("new-file.txt").write_text("New work\n")
        self.git("add", "new-file.txt")
        self.git("commit", "-qm", "New work")
        self.git("tag", "-f", "frozen-release")
        with self.assertRaisesRegex(ValueError, "Release tags changed"):
            preservation.verify()


if __name__ == "__main__":
    unittest.main()
