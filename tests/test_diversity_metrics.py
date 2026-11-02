import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

# The metric tests do not need the data reader or plotting backend.
reader_module = types.ModuleType('helpers.Reader')
reader_module.Reader = object
sys.modules['helpers.Reader'] = reader_module
matplotlib_module = types.ModuleType('matplotlib')
pyplot_module = types.ModuleType('matplotlib.pyplot')
matplotlib_module.pyplot = pyplot_module
sys.modules['matplotlib'] = matplotlib_module
sys.modules['matplotlib.pyplot'] = pyplot_module

from Inference import compute_diversity_metrics


CATALOG = np.arange(10)


class DiversityMetricsTests(unittest.TestCase):
    def test_partial_coverage_and_gini(self):
        # user 0 -> items 0,1 ; user 1 -> items 2,3
        predictions = [np.array([[0, 1]]), np.array([[2, 3]])]
        coverage, gini = compute_diversity_metrics(predictions, CATALOG, [2])
        self.assertEqual(coverage, [0.4])
        # counts: four items once, six never
        self.assertAlmostEqual(gini[0], 0.6, places=6)

    def test_uniform_recommendations_have_zero_gini(self):
        # both halves of the catalog recommended exactly once
        predictions = [
            np.array([[0, 1, 2, 3, 4]]),
            np.array([[5, 6, 7, 8, 9]]),
        ]
        coverage, gini = compute_diversity_metrics(predictions, CATALOG, [5])
        self.assertEqual(coverage, [1.0])
        self.assertAlmostEqual(gini[0], 0.0, places=6)

    def test_concentrated_recommendations_have_high_gini(self):
        # every user receives the same two items
        predictions = [np.array([[0, 1]]), np.array([[0, 1]])]
        coverage, gini = compute_diversity_metrics(predictions, CATALOG, [2])
        self.assertEqual(coverage, [0.2])
        self.assertAlmostEqual(gini[0], 0.8, places=6)

    def test_metrics_per_k(self):
        predictions = [np.array([[0, 1, 2, 3]]), np.array([[2, 1, 3, 4]])]
        coverage, gini = compute_diversity_metrics(predictions, CATALOG, [2, 4])
        self.assertEqual(coverage, [0.3, 0.5])
        # recommendation counts spread out as K grows -> gini strictly decreases
        self.assertGreater(gini[0], gini[1])


if __name__ == '__main__':
    unittest.main()
