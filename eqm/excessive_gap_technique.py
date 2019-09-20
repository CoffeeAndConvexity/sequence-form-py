import logging
import numpy as np

from .eqm import EquilibriumAlgorithm


class ExcessiveGapTechnique(EquilibriumAlgorithm):
    def __init__(self,
                 game,
                 prox_x=None,
                 prox_y=None,
                 L=1.0,
                 mu=2.0,
                 aggressive_stepsizes=True,
                 x=None,
                 y=None,
                 gradient_computations=0,
                 init_gap=0.0,
                 init_update_x=False,
                 allowed_eps_increase=-1.):
        EquilibriumAlgorithm.__init__(self, game)

        self._prox_x = prox_x if prox_x is not None else game.domain(0).prox()
        self._prox_y = prox_y if prox_y is not None else game.domain(1).prox()
        self._smooth_br_x = game.domain(0).smooth_br()
        self._smooth_br_y = game.domain(1).smooth_br()

        self._aggressive_stepsizes = aggressive_stepsizes
        self._init_gap = init_gap
        self._init_update_x = init_update_x
        self._allowed_eps_increase = allowed_eps_increase
        self._w = 1
        if aggressive_stepsizes:
            self._tau = 1.0 / 2.0
        else:
            self._tau = 2.0 / 3.0
        self._gradient_computations = gradient_computations
        if x is not None and y is not None:
            logging.info('warm-starting EGT')
            self._x, self._y = x, y
            self._mu = np.array([mu, mu * L])
            self.initial_step()
        elif self._init_gap < 0:
            L = game.payoff_max_norm()
            self._mu = np.array([L, L])
            self.initial_step()
        else:
            self._x = game.domain(0).prox().center()
            self.initial_step_search()

        assert self.excessive_gap() >= 0

    def initial_step_search(self):
        num_mu = 16
        self._gradient_computations += 2 * int(np.ceil(np.log2(num_mu))) * 2
        start = 0.001 / self._game.domain(0).diameter()
        for mu in np.logspace(np.log10(start), np.log10(1.0), num=num_mu):
            for lipschitz in [0.75, 1.0, 1.25]:
                mu2 = mu * lipschitz

                self._mu = np.array([mu, mu2])
                self.initial_step()
                gap = self.excessive_gap()
                logging.debug('mu=[%0.8f, %0.8f] gap=%0.04f', mu, mu2, gap)

                if gap > self._init_gap:
                    return

                if not self._init_update_x:
                    self._x = self._prox_x.center()
                # L = 2.0*L

    def initial_step(self):
        try:
            u_y = self._game.utility_for(1, self._x)
            _, self._y = self._smooth_br_y(-1.0, u_y, self._mu[1])

            u_x = self._game.utility_for(0, self._y)
            _, self._x = self._prox_x(-1.0, u_x, self._mu[0], self._x)

        except ValueError:
            pass

    def get_params_string(self):
        return 'val={:0.3f},\teps={:0.3f},\tmu=[{:0.4f}, {:0.4f}]\tgap={:0.4f}\tstep={:0.4f}'\
            .format(
                self.profile_value(),
                self.epsilon(), self._mu[0], self._mu[1],
                self.excessive_gap(), self._tau)

    def get_current_iterate_string(self):
        return np.array_str(
            self._x, max_line_width=999,
            suppress_small=True) + '\n' + np.array_str(
                self._y, max_line_width=999, suppress_small=True)

    def iterate(self, num_iterations=1):
        for _ in range(num_iterations):
            old_x = np.copy(self._x)
            old_y = np.copy(self._y)
            old_mu = np.copy(self._mu)
            self.shrink(self._tau, old_x, old_y)
            logging.debug(self.get_params_string())
            if self._aggressive_stepsizes:
                self._gradient_computations += 2
                while self.excessive_gap() < 0 or self.worse_than_old(
                        old_x, old_y, self._x, self._y):
                    self._gradient_computations += 2
                    self._mu = np.copy(old_mu)
                    self._tau *= 0.5
                    logging.getLogger().debug('%s, decreasing stepsize',
                                              self.get_params_string())
                    self.shrink(self._tau, old_x, old_y)
                if self._tau < 0.5:
                    self._tau *= 1.11
            else:
                self._tau = 2.0 / (self._w + 3)
            self._w += 1

    def worse_than_old(self, old_x, old_y, new_x, new_y):
        if self._allowed_eps_increase <= 1.0:
            return False
        old_eps, _, _, _ = self._game.profile_epsilon(old_x, old_y)
        new_eps, _, _, _ = self._game.profile_epsilon(new_x, new_y)
        return new_eps > old_eps * self._allowed_eps_increase

    def shrink(self, tau, x, y):
        # uncomment this line and comment the following line to do
        # traditional EGT rather than mu balancing.
        # if self._w % 2 == 1:  # shrink mu[0]
        if self._mu[0] > self._mu[1]:  # shrink mu[0]
            self._shrink_player(tau, x, y, 0, self._prox_x,
                                self._smooth_br_x, self._smooth_br_y)
        else:
            self._shrink_player(tau, y, x, 1, self._prox_y,
                                self._smooth_br_y, self._smooth_br_x)

    def _shrink_player(self, tau, x, y, player, prox_x, smooth_br_x,
                       smooth_br_y):
        opponent = 1 - player
        np.set_printoptions(precision=6)
        np.set_printoptions(suppress=True)
        # This gradient could be reused from last iteration's excessive gap
        # check
        u_x = self._game.utility_for(player, y)
        _, br_x = smooth_br_x(-1.0, u_x, self._mu[player])
        assert self._game.domain(player).is_behavioral_form(br_x)
        hat_x = self._game.domain(player).combine(x, tau, br_x)

        u_y = self._game.utility_for(opponent, hat_x)
        _, br_y = smooth_br_y(-1.0, u_y, self._mu[opponent])
        if player == 0:
            self._y = self._game.domain(opponent).combine(y, tau, br_y)
        else:
            self._x = self._game.domain(opponent).combine(y, tau, br_y)

        u_x = self._game.utility_for(player, br_y)
        assert self._game.domain(player).is_behavioral_form(br_x)
        _, br_x = prox_x(-tau, u_x, (1 - tau) * self._mu[player], br_x)
        if player == 0:
            self._x = self._game.domain(player).combine(x, tau, br_x)
        else:
            self._y = self._game.domain(player).combine(x, tau, br_x)

        self._mu[player] = (1 - 1.0 * tau) * self._mu[player]
        if self._aggressive_stepsizes:
            self._gradient_computations += 2
        else:
            self._gradient_computations += 3

    def excessive_gap(self):
        u_y = self._game.utility_for(1, self._x)
        val_f = -self._smooth_br_y(-1.0, u_y, self._mu[1])[0]

        u_x = self._game.utility_for(0, self._y)
        val_phi = self._smooth_br_x(-1.0, u_x, self._mu[0])[0]

        return val_phi - val_f

    def duality_gap_bound(self):
        return self._mu[0] * self._game.domain(0).diameter() +\
            self._mu[1] * self._game.domain(1).diameter()

    def fit_to_strategy(self):
        if self.excessive_gap() < 0:
            while self.excessive_gap() < 0:
                self._mu *= 2
                gap = self.excessive_gap()
                logging.info('mu=[%0.2f, %0.2f] gap=%0.04f', self._mu[0],
                             self._mu[1], gap)
        else:
            while self.excessive_gap() > 0:
                self._mu /= 2
                gap = self.excessive_gap()
                logging.info('mu=[%0.2f, %0.2f] gap=%0.04f', self._mu[0],
                             self._mu[1], gap)
            self._mu *= 2


def excessive_gap_technique_init(aggressive_stepsizes=False, **kwargs):
    def init(game):
        return ExcessiveGapTechnique(
            game, aggressive_stepsizes=aggressive_stepsizes, **kwargs)

    return init


def egt_warm_start_initializer(alg, aggressive_stepsizes=False, **kwargs):
    def init(game):
        opt = alg(game)
        opt.iterate(200)
        x, y = opt.profile()
        return ExcessiveGapTechnique(
            game,
            aggressive_stepsizes=aggressive_stepsizes,
            x=x,
            y=y,
            **kwargs)

    return init
