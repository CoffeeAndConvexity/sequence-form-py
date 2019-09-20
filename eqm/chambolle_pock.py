import numpy as np

from .eqm import EquilibriumAlgorithm

class ChambollePock(EquilibriumAlgorithm):
    def __init__(self, game, prox_x=None, prox_y=None, L=1.0):
        EquilibriumAlgorithm.__init__(self, game)

        self._prox_x = prox_x if prox_x is not None else game.domain(0).prox()
        self._prox_y = prox_y if prox_y is not None else game.domain(1).prox()

        self._x = self._prox_x.center()
        self._y = self._prox_y.center()

        self._c_x = np.copy(self._x)
        self._p_x = np.copy(self._x)
        self._c_y = np.copy(self._y)

        self._w = 1.0
        self._L = L

    def iterate(self, num_iterations=1):
        for t in range(num_iterations):
            u_py = self._game.utility_for(1, self._p_x)
            u_cy = self._game.utility_for(1, self._c_x)
            _, self._c_y = self._prox_y(-1.0, 2*u_cy - u_py, self._L, self._c_y)

            u_x = self._game.utility_for(0, self._c_y)
            _, c_x = self._prox_x(-1.0, u_x, self._L, self._c_x)

            self._p_x = self._c_x
            self._c_x = c_x
            
            alpha = 1.0/(1.0 + self._w); self._w += 1.0
            self._x = self._game.domain(0).combine(self._x, alpha, self._c_x)
            self._y = self._game.domain(1).combine(self._y, alpha, self._c_y)

            self._gradient_computations += 3
