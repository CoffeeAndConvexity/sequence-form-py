class EquilibriumAlgorithm:
    def __init__(self, game, name=None):
        self._game = game
        self._x    = game.domain(0).center()
        self._y    = game.domain(1).center()
        self._gradient_computations = 0

        self._name = name if name is not None else self.__class__.__name__

    def profile(self):
        return self._x, self._y

    def epsilon(self):
        eps, _, _, _ = self._game.profile_epsilon(self._x, self._y)
        return eps

    def profile_value(self):
        val = self._game.profile_value(self._x, self._y)
        return val

    def iterate(self, num_iterations=1):
        raise NotImplementedError

    def gradient_computations(self):
        return self._gradient_computations

    def __repr__(self):
        return self._name
