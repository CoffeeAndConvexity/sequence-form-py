import unittest
import numpy as np
from poker import kuhn
from extensive_form_game import treeplex
from matrix_game import simplex

def large_test_domain():
    """ Makes a small Treeplex for testing purposes. This treeplex has:
    1 Root infoset with 2 actions (1,2)
      Action 1 at root leads to 2 infosets, with actions (3,4) and (5,6,7)
        Action 5 leads to 2 infosets, with actions (8,9,10) and (11,12,13,14)
        Action 7 leads to 2 infosets, with actions (15,16) and (17,18,19)
      Action 2 at root leads to 1 infoset with 2 actions (20,21)

    """
    dimension = 22
    begin     = np.array([1, 3, 5, 8, 11, 15, 17, 20])
    end      = np.array([3, 5, 8, 11, 15, 17, 20, 22])
    parent    = np.array([0, 1, 1, 5, 5, 7, 7, 2])
    return treeplex.TreeplexDomain(dimension, begin, end, parent)

def small_test_domain():
    """ Makes a small Treeplex for testing purposes. This treeplex has:
    1 Root infoset with 2 actions (1,2)
    Action 0 at root leads to 2 infosets, each with 2 actions (3,4) and (5,6)
    Action 1 at root leads to 1 infoset with 2 actions (7,8)

    """
    dimension = 9
    begin     = np.array([1, 3, 5, 7])
    end      = np.array([3, 5, 7, 9])
    parent    = np.array([0, 1, 1, 2])
    return treeplex.TreeplexDomain(dimension, begin, end, parent)


class TestTreeplex(unittest.TestCase):
    def setUp(self):
        self.kuhn = kuhn.init_efg()
        self.small_treeplex = small_test_domain()
        self.large_treeplex = large_test_domain()
        self.p1_uniform_strat = self.kuhn.domain(0).center()
        self.p1_pure_strat = np.zeros(self.kuhn.domain(0).dimension())
        self.p1_pure_epsilon_strat = np.zeros(self.kuhn.domain(0).dimension())
        for idx in self.kuhn.domain(0)._begin:
            self.p1_pure_strat[idx] = 1
            self.p1_pure_epsilon_strat[idx] = 0.75
            self.p1_pure_epsilon_strat[idx + 1] = 0.25


    def tearDown(self):
        pass


    def test_infoset_regret_small_treeplex(self):
        uniform_strat = np.full(self.small_treeplex.dimension(),
                                   0.5, dtype=float)
        utility = np.array([
            0.0, #  empty sequence
            1.0, 0.0, #  infoset 0
            0.0, 1.0, #  infoset 1
            1.0, 0.0, #  infoset 2
            0.0, 1.0, #  infoset 3
        ])
        expected_regrets = np.array([
            0.75 + 0.75 + 0.5 - 0.25, #  infoset 0
            0.5, #  infoset 1
            0.5, #  infoset 2
            0.5, #  infoset 3
        ])
        regrets = self.small_treeplex.infoset_regrets(utility, uniform_strat)[0]
        assert np.allclose(regrets, expected_regrets)


    def test_infoset_regrets_kuhn(self):
        expected_regrets = np.full(self.kuhn.domain(0).num_information_sets(),
                                   50, dtype=float)
        for idx in range(3):
            expected_regrets[idx * 2] += 75
        expected_strat = np.zeros(self.kuhn.domain(0).dimension())
        expected_strat[0] = 1
        uniform_strat = np.full(self.kuhn.domain(0).dimension(),
                                0.5, dtype=float)
        uniform_strat[0] = 1
        g = np.zeros(self.kuhn.domain(0).dimension())
        for idx in self.kuhn.domain(0)._begin:
            expected_strat[idx + 1] = 1
            g[idx + 1] = 100
        regrets, strat = self.kuhn.domain(0).infoset_regrets(g, uniform_strat)
        assert np.array_equal(expected_strat, strat)
        assert np.array_equal(expected_regrets, regrets)

    def test_prox(self):
        prox_weight = 9.0
        ent = lambda x: prox_weight * (-entropy(x) + np.log(len(x)))
        smooth_f = lambda g,x: np.dot(g,x) + ent(x)
        beta = 0.5

        # test with no y
        g = np.array([0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0])
        val, z = self.small_treeplex.prox()(-1.0, g, beta)
        g = np.array([0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0])
        test_val, test_z = treeplex_prox_by_simplex_prox(self.small_treeplex, -1.0, g, beta)
        assert val == test_val
        assert np.allclose(z, test_z)

        # test with y
        g = np.array([0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0])
        y = np.array([0.0,0.8,0.2,0.8,0.2,0.8,0.2,0.8,0.2])
        val, z = self.small_treeplex.prox()(-1.0, g, beta, y)
        g = np.array([0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0])
        test_val, test_z = treeplex_prox_by_simplex_prox(self.small_treeplex, -1.0, g, beta, y)
        assert val == test_val
        assert np.allclose(z, test_z)

        for _ in range(20):
            g = np.random.rand(self.large_treeplex.dimension())
            g2 = g.copy()
            y = np.random.rand(self.large_treeplex.dimension())
            y[0] = 1
            for i in range(len(self.large_treeplex.prox()._weights)-1, -1, -1):
                begin = self.large_treeplex._begin[i]
                end = self.large_treeplex._end[i]
                parent = self.large_treeplex._parent[i]
                y[begin: end] /= sum(y[begin: end])

            val, z = self.large_treeplex.prox()(-1.0, g, beta, y)
            test_val, test_z = treeplex_prox_by_simplex_prox(self.large_treeplex, -1.0, g2, beta, y)
            assert np.allclose(z, test_z)
            assert abs(val - test_val) < 1e-5



def treeplex_prox_by_simplex_prox(tp, alpha, g, beta, y=None):
    z = np.zeros(tp.dimension())
    z[0] = 1.0
    g *= alpha
    for i in range(len(tp.prox()._weights)-1, -1, -1):
        begin = tp._begin[i]
        end = tp._end[i]
        parent = tp._parent[i]
        dgf_weight = beta * tp.prox()._weights[i]

        sprox = simplex.SimplexEntropyProx(end - begin)
        if y is not None:
            sval, sz = sprox(1.0, g[begin: end], dgf_weight, y[begin: end])
        else:
            sval, sz = sprox(1.0, g[begin: end], dgf_weight)
        z[begin: end] = sz
        g[parent] += sval

    return g[0], z

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTreeplex)
    unittest.TextTestRunner(verbosity=2).run(suite)
