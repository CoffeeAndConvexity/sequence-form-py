import unittest
from test_extensive_form_game import TestExtensiveFormGame
from test_kuhn import TestKuhn
from test_leduc import TestLeduc
from test_simplex import TestSimplex
from test_treeplex import TestTreeplex

if __name__ == '__main__':
    alltests = unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(TestKuhn),
        unittest.TestLoader().loadTestsFromTestCase(TestLeduc),
        unittest.TestLoader().loadTestsFromTestCase(TestSimplex),
        unittest.TestLoader().loadTestsFromTestCase(TestTreeplex),
        unittest.TestLoader().loadTestsFromTestCase(TestExtensiveFormGame),
    ])
    unittest.TextTestRunner(verbosity=2).run(alltests)
