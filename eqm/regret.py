import math
import numpy as np

from .eqm import EquilibriumAlgorithm
from extensive_form_game.cfr import CounterfactualRegretMinimizer
from extensive_form_game.treeplex import TreeplexDomain
"""
returns the sequence {alpha + beta*sqrt(t) + gamma*t}_{t=1}^\inf
"""


def step_size_generator(alpha, beta, gamma):
    t = 1
    while True:
        t += 1
        yield alpha + beta * math.sqrt(t - 1) + gamma * (t - 1)


class RegretMinimization(EquilibriumAlgorithm):
    def __init__(self,
                 game,
                 rm_x,
                 rm_y=None,
                 alternate=False,
                 step=step_size_generator(1.0, 0.0, 0.0),
                 name=None):
        def _init_rm(domain, rm):
            if isinstance(domain, TreeplexDomain):
                regret_matcher = CounterfactualRegretMinimizer(
                    domain, rm, name)
            else:
                regret_matcher = rm(domain)
            self._name = str(regret_matcher)
            return regret_matcher

        rm_y = rm_x if rm_y is None else rm_y
        self._rm_x = _init_rm(game.domain(0), rm_x)
        self._rm_y = _init_rm(game.domain(1), rm_y)
        self._alternate = alternate

        EquilibriumAlgorithm.__init__(self, game, name=self._name)

        self._step = step
        self._alpha = next(step)
        self._weight = self._alpha

    def iterate(self, num_iterations=1):
        for t in range(num_iterations):
            u_x = self._game.utility_for(0, self._rm_y.strategy)
            if not self._alternate:
                u_y = self._game.utility_for(1, self._rm_x.strategy)

            self._gradient_computations += 2

            self._rm_x(u_x)

            if self._alternate:
                u_y = self._game.utility_for(1, self._rm_x.strategy)

            self._rm_y(u_y)

            self._alpha = next(self._step)
            self._weight += self._alpha
            alpha = self._alpha / self._weight
            self._x = self._game.domain(0).combine(self._x, alpha,
                                                   self._rm_x.strategy)
            self._y = self._game.domain(1).combine(self._y, alpha,
                                                   self._rm_y.strategy)


def regret_minimization_initializer(rm_x,
                                    rm_y=None,
                                    linear_averaging=False,
                                    **kwargs):
    def init(game, name=None):
        if linear_averaging:
            step = step_size_generator(1.0, 0.0, 1.0)
        else:
            step = step_size_generator(1.0, 0.0, 0.0)
        return RegretMinimization(game, rm_x, rm_y, step=step, **kwargs)

    return init
