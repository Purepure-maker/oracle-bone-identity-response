"""Public command interfaces must preserve self exclusion and mapping identity."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,sys,tempfile,unittest,subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('public_image_cli',ROOT/'reproduce_images.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

class ReleaseInterfaceTests(unittest.TestCase):
    def bundle(self,folder,keys,features,mapping='same_mapping',classes=None):
        folder.mkdir()
        np.save(folder/'ID.npy',np.asarray(features,np.float32))
        (folder/'paths.json').write_text(json.dumps(keys),encoding='utf8')
        (folder/'class_order.json').write_text(json.dumps(classes or ['a','b']),encoding='utf8')
        metadata={'state':'COMPLETE','model_state_sha256':'frozen_model','mapping_sha256':mapping,'arrays_sha256':{'ID.npy':cli.sha(folder/'ID.npy')}}
        (folder/'complete.json').write_text(json.dumps(metadata),encoding='utf8')
        return folder

    def test_independent_query_and_same_key_self_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            gallery=self.bundle(root/'gallery',['a/self','a/other','b/other'],[[1,0],[.99,.01],[0,1]])
            query=self.bundle(root/'query',['a/self','unlabeled_query'],[[1,0],[0,1]])
            target=root/'rank.json'
            cli.retrieve(SimpleNamespace(queries=query,gallery=gallery,top_k=10,output=target))
            rows=json.loads(target.read_text(encoding='utf8'))['queries']
            self.assertEqual(rows[0]['candidates'][0]['key'],'a/other')
            self.assertNotIn('a/self',[c['key'] for c in rows[0]['candidates']])
            self.assertEqual(rows[1]['candidates'][0]['key'],'b/other')

    def test_incompatible_calibration_or_prototypes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            gallery=self.bundle(root/'gallery',['a/0','b/0'],[[1,0],[0,1]])
            query=self.bundle(root/'query',['query'],[[1,0]],mapping='changed_prototypes')
            with self.assertRaisesRegex(AssertionError,'mapping differs'):
                cli.retrieve(SimpleNamespace(queries=query,gallery=gallery,top_k=1,output=root/'rank.json'))

    def test_corrupted_descriptor_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            gallery=self.bundle(root/'gallery',['a/0','b/0'],[[1,0],[0,1]])
            query=self.bundle(root/'query',['query'],[[1,0]])
            np.save(gallery/'ID.npy',np.array([[0,1],[1,0]],np.float32))
            with self.assertRaisesRegex(AssertionError,'hash mismatch'):
                cli.retrieve(SimpleNamespace(queries=query,gallery=gallery,top_k=1,output=root/'rank.json'))

    def test_cli_help_does_not_require_torch_or_assets(self):
        run=subprocess.run([sys.executable,str(ROOT/'reproduce_images.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn('retrieve',run.stdout)
        self.assertIn('evaluate',run.stdout)

if __name__=='__main__':unittest.main()
