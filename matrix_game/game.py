import numpy as np
from .simplex import SimplexDomain

class MatrixGame:
    """
    represents the saddle-point problem:
    min_{x\in\Delta} max_{y\in\Delta} x'Ay

    i.e., x, the first player, is the minimizer
    """
    def __init__(self, name, A):
        self._name    = name
        self._A       = A
        self._domains = (SimplexDomain(A.shape[0]), SimplexDomain(A.shape[1]))

    def domain(self, player):
        return self._domains[player]

    def profile_epsilon(self, x, y):
        value   = self.profile_value(x, y)
        br_x, _ = self.domain(0).support(self.utility_for(0, y))
        br_y, _ = self.domain(1).support(self.utility_for(1, x))
        return br_x + br_y, br_x - value, br_y + value, value

    def profile_value(self, x, y):
        return np.dot(x, self.utility_for(0, y))

    def utility_for(self, player, opponent_strategy):
        if player == 0:
            return -np.dot(self._A, opponent_strategy)

        assert player == 1
        return np.dot(self._A.T, opponent_strategy)

    def __str__(self):
        return 'MatrixGame(%s, %dx%d)' % (self._name, self._A.shape[0], self._A.shape[1])
