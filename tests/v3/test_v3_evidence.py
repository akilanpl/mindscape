import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("v3_restore", ROOT / "scripts/v3/restore_evidence.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvidenceSecurity(unittest.TestCase):
    def test_saved_evidence_clean_restore_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = module.restore(
                ROOT / "experiments/v3/cpu-smoke.tar.gz",
                ROOT / "experiments/v3/evidence_manifest.json",
                Path(tmp, "restore"),
            )
            self.assertEqual(result["replayed_executions"], 450)
            self.assertEqual(result["neural_forward_calls"], 0)

    def test_archive_corruption_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp, "bad.tar.gz")
            archive.write_bytes(b"bad archive")
            with self.assertRaises(ValueError):
                module.restore(
                    archive, ROOT / "experiments/v3/evidence_manifest.json", Path(tmp, "restore")
                )
            self.assertFalse(Path(tmp, "restore").exists())

    def test_traversal_rejected_even_with_matching_outer_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, manifest = Path(tmp, "bad.tar.gz"), Path(tmp, "manifest.json")
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("../escape")
                info.size = 1
                tar.addfile(info, io.BytesIO(b"x"))
            manifest.write_text(
                json.dumps(
                    {
                        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                        "files": {"../escape": hashlib.sha256(b"x").hexdigest()},
                    }
                )
            )
            with self.assertRaises(ValueError):
                module.restore(archive, manifest, Path(tmp, "restore"))
            self.assertFalse(Path(tmp, "restore").exists())


if __name__ == "__main__":
    unittest.main()
