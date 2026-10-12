import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "coding_evidence", ROOT / "scripts/v3_coding/evidence.py"
)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class CodingEvidenceTests(unittest.TestCase):
    def test_multipart_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "raw")
            root.mkdir()
            (root / "file.json").write_text('{"observed":true}')
            package = Path(tmp, "package")
            destination = Path(tmp, "restored")
            evidence.pack(root, package)
            result = evidence.restore(package, destination)
            self.assertEqual(result["verified_files"], 1)
            self.assertEqual(
                (root / "file.json").read_bytes(), (destination / "file.json").read_bytes()
            )

    def test_corruption_rejected_before_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "raw")
            root.mkdir()
            (root / "file").write_text("observed")
            package = Path(tmp, "package")
            destination = Path(tmp, "restored")
            evidence.pack(root, package)
            next(package.glob("*.part*")).write_bytes(b"bad")
            with self.assertRaises(ValueError):
                evidence.restore(package, destination)
            self.assertFalse(destination.exists())

    def test_traversal_rejected_with_matching_part_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp, "package")
            package.mkdir()
            part = package / "evidence.part000"
            with tarfile.open(part, "w:gz") as tar:
                info = tarfile.TarInfo("../escape")
                info.size = 1
                tar.addfile(info, io.BytesIO(b"x"))
            sha = hashlib.sha256(part.read_bytes()).hexdigest()
            (package / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "mindscape.v3.coding.evidence/1",
                        "archive_sha256": sha,
                        "files": {"../escape": hashlib.sha256(b"x").hexdigest()},
                        "parts": [{"name": part.name, "sha256": sha, "size": part.stat().st_size}],
                    }
                )
            )
            destination = Path(tmp, "restore")
            with self.assertRaises(ValueError):
                evidence.restore(package, destination)
            self.assertFalse(destination.exists())
            self.assertFalse(Path(tmp, "escape").exists())
