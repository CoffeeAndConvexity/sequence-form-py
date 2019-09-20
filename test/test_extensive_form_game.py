import unittest
import numpy as np
from poker import kuhn

class TestExtensiveFormGame(unittest.TestCase):
    def setUp(self):
        self.kuhn = kuhn.init_efg()
        self.p1_uniform_strat = self.kuhn.domain(0).center()
        self.p1_pure_strat = np.zeros(self.kuhn.domain(0).dimension())
        self.p1_pure_strat[0] = 1
        self.p2_uniform_strat = self.kuhn.domain(1).center()
        self.p2_pure_strat = np.zeros(self.kuhn.domain(1).dimension())
        self.p2_pure_strat[0] = 1
        for idx in self.kuhn.domain(0)._begin:
            self.p1_pure_strat[idx] = 1
        for idx in self.kuhn.domain(1)._begin:
            self.p2_pure_strat[idx] = 1

    def tearDown(self):
        pass

    def test_reach(self):
        pure_expected_reach_p1 = np.zeros(3 * 2) # 3 hands, 2 infosets per hand
        pure_expected_reach_p2 = np.zeros(3 * 2)
        uniform_expected_reach_p1 = np.zeros(3 * 2)
        uniform_expected_reach_p2 = np.zeros(3 * 2)
        for idx in range(3):
            pure_expected_reach_p1[idx * 2] = 1.0 / 3
            pure_expected_reach_p1[idx * 2 + 1] = 1.0 / 3
            pure_expected_reach_p2[idx * 2] = 1.0 / 3
            pure_expected_reach_p2[idx * 2 + 1] = 0
            uniform_expected_reach_p1[idx * 2] = 1.0 / 3
            uniform_expected_reach_p1[idx * 2 + 1] = 0.5 / 3
            uniform_expected_reach_p2[idx * 2] = 0.5 / 3
            uniform_expected_reach_p2[idx * 2 + 1] = 0.5 / 3

        pure_reach_p1 = self.kuhn.reach(0, self.p2_pure_strat)
        assert np.array_equal(pure_expected_reach_p1, pure_reach_p1)
        pure_reach_p2 = self.kuhn.reach(1, self.p1_pure_strat)
        assert np.array_equal(pure_expected_reach_p2, pure_reach_p2)
        uniform_reach_p1 = self.kuhn.reach(0, self.p2_uniform_strat)
        assert np.array_equal(uniform_expected_reach_p1, uniform_reach_p1)
        uniform_reach_p2 = self.kuhn.reach(1, self.p1_uniform_strat)
        assert np.array_equal(uniform_expected_reach_p2, uniform_reach_p2)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExtensiveFormGame)
    unittest.TextTestRunner(verbosity=2).run(suite)
