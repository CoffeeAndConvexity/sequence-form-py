import unittest
import numpy as np
from scipy.stats import entropy
from matrix_game import simplex

class TestSimplex(unittest.TestCase):
    def setUp(self):
        self.prox = simplex.SimplexEntropyProx(5)

    def tearDown(self):
        pass

    def test_prox_value(self):
        """ Test that entropy prox value is computed correctly by computing
        the prox mapping and checking that its prox value corresponds to the
        value computed by the simplex prox function
        """
        prox_weight = 9.0
        ent = lambda x: prox_weight * (-entropy(x) + np.log(len(x)))
        smooth_f = lambda g,x: np.dot(g,x) + ent(x)
        for _ in range(20):
            gradient = np.random.rand(5) #  random gradient example
            prox_val, prox_map = self.prox(1.0, gradient, prox_weight)
            assert np.isclose(smooth_f(gradient, prox_map), prox_val)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSimplex)
    unittest.TextTestRunner(verbosity=2).run(suite)
