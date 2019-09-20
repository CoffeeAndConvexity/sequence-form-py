import numpy as np


class Hedge:
    def __init__(self, dimension, alpha):
        self._dimension = dimension
        self._alpha = alpha
        self.strategy = np.ones(dimension) / dimension
        self.regret = np.zeros(dimension)

    def __call__(self, utility):
        value = np.dot(self.strategy, utility)

        self.regret += utility
        self.regret -= value

        offset = np.max(self.regret)
        np.exp(self._alpha * (self.regret - offset), out=self.strategy)

        Z = np.sum(self.strategy)
        self.strategy /= Z

        return value

    def __str__(self):
        return 'Hedge(%f)' % self._alpha


class RegretMatching:
    def __init__(self, dimension):
        self._dimension = dimension
        self.strategy = np.ones(dimension) / dimension
        self.regret = np.zeros(dimension)

    def __call__(self, utility):
        value = np.dot(self.strategy, utility)

        self.regret += utility
        self.regret -= value

        np.maximum(self.regret, 0, out=self.strategy)

        Z = np.sum(self.strategy)
        if Z <= 0.0:
            self.strategy.fill(1.0)
            Z = self._dimension

        self.strategy /= Z

        return value

    def __str__(self):
        return 'RegretMatching'


class RegretMatchingPlus:
    def __init__(self, dimension):
        self._dimension = dimension
        self.strategy = np.ones(dimension) / dimension
        self.regret = np.zeros(dimension)

    def __call__(self, utility):
        value = np.dot(self.strategy, utility)

        self.regret += utility
        self.regret -= value

        np.maximum(self.regret, 0, out=self.strategy)
        np.maximum(self.regret, 0, out=self.regret)

        Z = np.sum(self.strategy)
        if Z <= 0.0:
            self.strategy.fill(1.0)
            Z = self._dimension

        self.strategy /= Z

        return value

    def __str__(self):
        return 'RegretMatching+'


def regret_matching_bound(dimension, payoff, num_iterations):
    return payoff * np.sqrt(dimension * num_iterations)


def hedge_initializer(alpha=1.0):
    def init(domain):
        return Hedge(domain.dimension(), alpha)

    return init


def regret_matching_initializer():
    def init(domain):
        return RegretMatching(domain.dimension())

    return init


def regret_matching_plus_initializer():
    def init(domain):
        return RegretMatchingPlus(domain.dimension())

    return init
