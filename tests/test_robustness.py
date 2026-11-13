"""Controls for altered-query self exclusion, ties and deterministic panel selection."""
from pathlib import Path
import itertools
import unittest
import numpy as np
from robustness.rectangular_evaluation import score, expected_scores
from robustness.run_study import select_indices


class RobustnessTests(unittest.TestCase):
    def test_transformed_origin_is_excluded(self):
        metrics,rows,_,_ = score(np.array([[.99,.01]]),
            np.array([[1.,0.],[.8,.2],[0.,1.]]),
            ['a/self'],['a/self','a/positive','b/negative'],workers=1)
        self.assertEqual(rows[0]['top5'],['a/positive','b/negative'])
        self.assertEqual(metrics['Class_AP'],1.)

    def test_exact_tie_matches_exhaustive_expectation(self):
        values = []
        for order in set(itertools.permutations([1,1,0,0])):
            ranks = np.flatnonzero(order)+1
            values.append(np.mean(np.arange(1,3)/ranks))
        result = expected_scores(np.zeros(4),np.array([1,1,0,0],bool))
        self.assertAlmostEqual(result['AP'],np.mean(values),places=14)
        self.assertEqual(result['Hit1'],.5)

    def test_panel_selection_is_key_based_and_excludes_singletons(self):
        keys = ['a/0','a/1','a/2','b/0','b/1','single/0']
        selected = {keys[i] for i in select_indices(keys,cap=1)}
        reversed_keys = list(reversed(keys))
        self.assertEqual(selected,{reversed_keys[i] for i in select_indices(reversed_keys,cap=1)})
        self.assertEqual(len(selected),2)
        self.assertNotIn('single/0',selected)

    def test_margins_keep_original_pixels_and_native_resize(self):
        try:
            from robustness.corruptions import apply
        except ImportError:
            self.skipTest('OpenCV is installed by the image requirements and hosted CI')
        image = np.arange(19*31*3,dtype=np.uint8).reshape(19,31,3)
        changed = apply(image,{'family':'asymmetric_margin','severity':.2})
        np.testing.assert_array_equal(changed[4:,6:],image)
        self.assertEqual(apply(image,{'family':'downsample','severity':.5}).shape,(10,16,3))


if __name__ == '__main__':
    unittest.main()
