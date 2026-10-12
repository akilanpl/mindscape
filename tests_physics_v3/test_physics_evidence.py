"""Physics evidence packaging verifies bytes and rejects unsafe archive inputs."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('physics_evidence', 'scripts/v3_physics/evidence.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class EvidenceTests(unittest.TestCase):
    def test_roundtrip_and_corrupt_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            (source / 'trace.json').write_text('{"device":"mps"}')
            package = root / 'package'
            evidence.pack(source, package)
            evidence.restore(package, root / 'restored')
            self.assertEqual((source / 'trace.json').read_bytes(),
                             (root / 'restored/trace.json').read_bytes())
            part = package / 'evidence.part000'
            part.write_bytes(part.read_bytes() + b'corrupt')
            with self.assertRaises(ValueError):
                evidence.restore(package, root / 'rejected')
            self.assertFalse((root / 'rejected').exists())

    def test_reject_unsafe_part_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            (source / 'trace').write_text('evidence')
            package = root / 'package'
            evidence.pack(source, package)
            manifest = json.loads((package / 'manifest.json').read_text())
            manifest['parts'][0]['name'] = '../escape'
            (package / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                evidence.restore(package, root / 'rejected')
            (source / 'link').symlink_to(source / 'trace')
            with self.assertRaises(ValueError):
                evidence.pack(source, root / 'unsafe-package')
