import importlib.util
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).resolve().parents[2]/'research/alpha/sprint_scope_audit.py'
spec=importlib.util.spec_from_file_location('scope_audit',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ScopeTests(unittest.TestCase):
    def test_tampering_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();(root/'source').write_bytes(b'original')
            h=m.sha(b'original')
            self.assertEqual(m.read_verified(root,'source',h),b'original')
            for name in ('../source','/source'):
                with self.assertRaises(ValueError):m.read_verified(root,name,h)
            (root/'source').write_bytes(b'changed')
            with self.assertRaises(ValueError):m.read_verified(root,'source',h)
    def test_symlink_member_and_ancestor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();(root/'real').mkdir();(root/'real/f').write_bytes(b'x')
            (root/'alias').symlink_to(root/'real',target_is_directory=True)
            with self.assertRaises(ValueError):m.read_verified(root,'alias/f',m.sha(b'x'))
    def test_full_frozen_identity(self):
        self.assertEqual(len(m.PINS),3)
        self.assertEqual(len(m.LOCKS),3)
        self.assertTrue(all(len(v[1])==64 for v in m.PROTOCOLS.values()))

if __name__=='__main__':unittest.main()
