from pathlib import Path
import itertools,json,sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduce import prototypes,responses,normalize,summary
from geometry import fixed_global_features
from evaluation import evaluate

class ReproductionTests(unittest.TestCase):
    def test_prototype_class_codes_and_float32_accumulation(self):
        x=np.array([[1,0],[0,1],[1,1]],np.float32)
        means,order,counts=prototypes(x,['0001','0002','0001'])
        self.assertEqual(order.tolist(),['0001','0002']);self.assertEqual(counts.tolist(),[2,1])
        np.testing.assert_allclose(means[0],np.array([2,1])/np.sqrt(5),atol=1e-7)
        np.testing.assert_array_equal(means[1],[0,1])

    def test_response_geometry_and_coordinate_permutation(self):
        rng=np.random.default_rng(12)
        x=normalize(rng.normal(size=(9,7))).astype(np.float32)
        p=normalize(rng.normal(size=(5,7))).astype(np.float32)
        z=responses(x,p);perm=[4,0,3,2,1];zp=responses(x,p[perm])
        np.testing.assert_allclose(np.sum(z*z,1),1,atol=2e-7)
        np.testing.assert_allclose(zp,z[:,perm],atol=2e-7)
        np.testing.assert_allclose(z@z.T,zp@zp.T,atol=3e-7)
        np.testing.assert_allclose(np.sum((z[0]-z[1])**2),2-2*np.dot(z[0],z[1]),atol=3e-7)

    def test_global_map_preserves_auxiliary_direction_and_shares(self):
        x=np.array([[.3,.4]+[0.]*510+[.4,.3]+[0.]*126],dtype=np.float32)
        center=np.zeros(640);center[0]=.1;w=np.eye(640);w[0,0]=2
        actual=fixed_global_features(x,center,w)
        global_expected=(np.array([.3,.4])-np.array([.1,0]))*np.array([2,1])
        global_expected=global_expected/np.linalg.norm(global_expected)*np.sqrt(.75)
        np.testing.assert_allclose(actual[0,:2],global_expected,atol=1e-7)
        np.testing.assert_allclose(actual[0,512:514],[.4,.3],atol=1e-7)
        np.testing.assert_allclose(np.sum(actual[:,:512]**2),.75,atol=1e-7)
        np.testing.assert_allclose(np.sum(actual[:,512:]**2),.25,atol=1e-7)

    def test_exact_tie_AP_matches_exhaustive_random_order_expectation(self):
        # Four remaining candidates, two relevant, after self exclusion.
        x=np.ones((5,2),np.float32);keys=['a/q0','a/q1','a/q2','b/q0','b/q1']
        ap=[]
        for order in itertools.permutations([1,1,0,0]):
            positions=np.flatnonzero(order)+1
            ap.append(np.mean(np.arange(1,3)/positions))
        with tempfile.TemporaryDirectory() as d:
            metrics,rows=evaluate(x,keys,Path(d)/'ties',workers=1)
        self.assertAlmostEqual(rows[0]['AP'],float(np.mean(ap)),places=12)
        self.assertAlmostEqual(rows[0]['Hit1'],.5,places=12)
        self.assertEqual(metrics['queries'],5)

    def test_singleton_is_negative_candidate_but_not_query(self):
        x=np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[1,1]],np.float32)
        keys=['a/0','a/1','a/2','b/0','b/1','c/0']
        with tempfile.TemporaryDirectory() as d:
            metrics,rows=evaluate(x,keys,Path(d)/'singletons',workers=1)
        self.assertEqual(metrics['gallery_images'],6);self.assertEqual(metrics['queries'],5)
        self.assertEqual(metrics['classes'],2);self.assertEqual(metrics['excluded_no_positive_queries'],1)
        self.assertAlmostEqual(metrics['Query_AP'],1.);self.assertAlmostEqual(metrics['Class_AP'],1.)

    def test_released_evidence_reconstructs_paper_summary(self):
        s=summary(Path(__file__).resolve().parents[1]/'evidence')
        self.assertEqual(s['supports'],{'identities':277,'images':17329})
        self.assertAlmostEqual(s['Query_delta_pp']['mean'],4.086582232610818,places=10)
        self.assertAlmostEqual(s['Class_delta_pp']['mean'],12.631258753460784,places=10)
        self.assertAlmostEqual(s['Class_delta_pp']['sample_SD'],.2120788072366738,places=10)

if __name__=='__main__':unittest.main()
