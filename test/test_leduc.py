import unittest
import numpy as np
from poker import leduc

class TestLeduc(unittest.TestCase):
    def setUp(self):
        self.leduc = leduc.init_efg()
        self.tolerance = 0.000001
        self.p1_uniform_strat = self.leduc.domain(0).center()
        self.p2_uniform_strat = self.leduc.domain(1).center()

    def test_sequences_length(self):
        assert len(self.leduc.domain(0).center()) == 337
        assert len(self.leduc.domain(1).center()) == 337

    def test_uniform_reach_p1(self):
        reach = self.leduc.reach(0, self.p2_uniform_strat)
        assert abs(reach[0] - 1.0 / 3) < self.tolerance
        assert abs(reach[1] - 1.0 / 3) < self.tolerance
        assert abs(reach[2] - 1.0 / 3) < self.tolerance

        # total combinations = 120
        # ways to match board for specific card = 2
        # ways to not match board = 4
        # possible opponent cards = 4
        reach_match_board = 8.0 / 120.0
        reach_diff_board = 16.0 / 120.0
        assert abs(reach[3] - reach_match_board * 0.5) < self.tolerance
        assert abs(reach[4] - reach_diff_board * 0.5) < self.tolerance
        assert abs(reach[5] - reach_diff_board * 0.5) < self.tolerance


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLeduc)
    unittest.TextTestRunner(verbosity=2).run(suite)
