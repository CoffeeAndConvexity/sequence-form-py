import unittest
import numpy as np
from poker import kuhn

class TestKuhn(unittest.TestCase):
    def setUp(self):
        self.kuhn = kuhn.init_efg()
        self.tolerance = 0.000001
        self.p1_uniform_strat = self.kuhn.domain(0).center()
        self.p1_pure_strat = np.zeros(self.kuhn.domain(0).dimension())
        for idx in self.kuhn.domain(0)._begin:
            self.p1_pure_strat[idx] = 1

    def test_sequences_length(self):
        assert len(self.kuhn.domain(0).center()) == 3 * 4 + 1
        assert len(self.kuhn.domain(1).center()) == 3 * 4 + 1

    def test_equilibrium_value(self):
        strategy_p1 = np.array([1,
                                0.333333, 0.666666, 0, 1,
                                0, 1, 0.666666, 0.333333,
                                1, 0, 1, 0
        ])
        strategy_p2 = np.array([1,
                                0, 1, 0.333333, 0.666666,
                                0.333333, 0.666666, 0, 1,
                                1, 0, 1, 0
        ])
        assert abs(self.kuhn.profile_value(strategy_p1, strategy_p2) + 0.0555555555) < self.tolerance

    def test_raise_call_value(self):
        strategy_p1 = np.array([ 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ])
        strategy_p2 = np.array([ 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ])
        assert abs(self.kuhn.profile_value(strategy_p1, strategy_p2)) < self.tolerance


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKuhn)
    unittest.TextTestRunner(verbosity=2).run(suite)
