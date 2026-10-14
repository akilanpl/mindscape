"""Independent Git boundary review: temporary repositories, no scientific imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('independent_scope_review', REPO/'research/alpha/sprint_scope_audit.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class GitScopeReview(unittest.TestCase):
    def git(self, *args):
        return subprocess.check_output(['git','-C',str(self.root),*args],stderr=subprocess.DEVNULL).decode().strip()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()/'repo'
        self.root.mkdir()
        self.git('init','-q')
        self.git('config','user.email','fixture@example.test')
        self.git('config','user.name','Synthetic Fixture')
        protocol = {'fits':6,'updates_per_fit':800,'batch_size':32,'seeds':[11,23,47],
                    'counts':{'train':1024},'gates_per_seed_per_model':{'unchanged_fixture_gate':0.95}}
        self.data = json.dumps(protocol).encode()
        (self.root/'protocol.json').write_bytes(self.data)
        (self.root/'frozen.py').write_text('frozen fixture\n')
        self.git('add','.')
        self.git('commit','-qm','baseline fixture')
        self.pin = self.git('rev-parse','HEAD')
        for mapping,values in [(m.PINS,{'r54':self.pin}),
                               (m.PROTOCOLS,{'r54':('protocol.json',m.sha(self.data))})]:
            manager = patch.dict(mapping,values)
            manager.start(); self.addCleanup(manager.stop)
        manager = patch.dict(m.LOCKS,{},clear=True)
        manager.start(); self.addCleanup(manager.stop)

    def audit(self):
        return m.audit_one(self.root,'r54')

    def test_baseline_and_addition_digest(self):
        self.assertEqual(self.audit()['baseline_files_verified'],2)
        (self.root/'candidate.py').write_text('new candidate\n')
        self.git('add','.')
        self.git('commit','-qm','add fixture')
        receipt = self.audit()
        self.assertEqual(receipt['candidate_file_sha256']['candidate.py'],m.sha(b'new candidate\n'))

    def test_dirty_baseline_rejected(self):
        (self.root/'frozen.py').write_text('tampered\n')
        with self.assertRaises(ValueError): self.audit()

    def test_committed_baseline_change_rejected(self):
        (self.root/'frozen.py').write_text('tampered\n')
        self.git('add','.')
        self.git('commit','-qm','change fixture')
        with self.assertRaises(ValueError): self.audit()

    def test_dirty_added_candidate_rejected(self):
        (self.root/'candidate.py').write_text('new\n')
        self.git('add','.')
        self.git('commit','-qm','add fixture')
        (self.root/'candidate.py').write_text('tampered\n')
        with self.assertRaises(ValueError): self.audit()

    def test_untracked_candidate_rejected(self):
        (self.root/'untracked.py').write_text('unreviewed\n')
        with self.assertRaises(ValueError): self.audit()

    def test_wrong_ancestry_rejected(self):
        with patch.dict(m.PINS,{'r54':'0'*40}):
            with self.assertRaises(ValueError): self.audit()

    def test_symlink_checkout_rejected(self):
        link = self.root.parent/'linked'
        link.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError): m.audit_one(link,'r54')

if __name__ == '__main__': unittest.main()
