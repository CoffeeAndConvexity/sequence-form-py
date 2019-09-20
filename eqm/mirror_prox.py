import math
import logging
import numpy as np

from .eqm import EquilibriumAlgorithm


class MirrorProx(EquilibriumAlgorithm):
    def __init__(self,
                 game,
                 prox_x=None,
                 prox_y=None,
                 num_fixed_point_iterations=2,
                 aggressive_stepsizes=True):
        EquilibriumAlgorithm.__init__(self, game, name="MirrorProx(AS)"
                                      if aggressive_stepsizes else "MirrorProx")

        self._prox_x = prox_x if prox_x is not None else game.domain(0).prox()
        self._prox_y = prox_y if prox_y is not None else game.domain(1).prox()

        self._x = self._prox_x.center()
        self._y = self._prox_y.center()

        self._c_x = np.copy(self._x)
        self._c_y = np.copy(self._y)

        # According to Nemirovski04, we can set parameters optimally as
        # M_kl = L_kl * sqrt(Omega_k * Omega_l / sigma_k * sigma_l)
        # \alpha_k = \sum_l M_kl / \sum_pl M_pl
        # \beta_k = \alpha_k / \Omega_k
        #
        # However, in our setting we just have 2 indices 1,2 and L_12=L_21
        # So M_12 = M_21. Thus \alpha_1 = \alpha_2 = 1/2, and
        # \beta_k = 1 / (2 * \Omega_k)
        #
        # Thus we get the DGF \beta_1 d_1 + \beta_2 d_2
        self._prox_weights = np.array(  # [1., 1.])
            [0.5 / game.domain(0).diameter(), 0.5 / game.domain(1).diameter()])

        # With the above setup, the Lipschitz constant is
        # L = 2 |A|_max * sqrt(diameter_1 * diameter_2)
        self._gamma_safe = 0.5 / (game.payoff_max_norm() *
                                  np.sqrt(game.domain(0).diameter() * game.domain(1).diameter()))
        self._gamma = 1.0 * self._gamma_safe
        self._w = 0.0

        self._aggressive_stepsizes = aggressive_stepsizes

    def iterate(self, num_iterations=2):
        for _ in range(num_iterations):
            self.take_step()

    def take_step(self):
        next_w_x = self._c_x
        next_w_y = self._c_y
        fixed_point_iters = 0
        while ((not self._aggressive_stepsizes) and fixed_point_iters < 2) or \
            (self._aggressive_stepsizes and self.stepsize_condition_violated(
                self._gamma, next_w_x, next_w_y, cur_w_x, cur_w_y, self._c_x,
                self._c_y)):
            # if we have done three iterations then we were too aggressive
            if fixed_point_iters > 2:
                logging.debug("fixed-point iterations: {}, gamma: {}".format(
                    fixed_point_iters, self._gamma))
                self._gamma = max(self._gamma_safe, self._gamma / 2.0)
            cur_w_x = next_w_x
            cur_w_y = next_w_y
            u_x = self._game.utility_for(0, cur_w_y)
            u_y = self._game.utility_for(1, cur_w_x)

            _, next_w_x = self._prox_x(-self._gamma, u_x,
                                       self._prox_weights[0], self._c_x)
            _, next_w_y = self._prox_y(-self._gamma, u_y,
                                       self._prox_weights[1], self._c_y)

            self._gradient_computations += 2
            fixed_point_iters += 1
            print("Fixed-point iters: %s" % fixed_point_iters)
            self.stepsize_condition_violated(self._gamma, next_w_x, next_w_y,
                                             cur_w_x, cur_w_y, self._c_x,
                                             self._c_y)

        self._c_x = next_w_x
        self._c_y = next_w_y

        self._w += self._gamma
        gamma = self._gamma / self._w
        self._x = self._game.domain(0).combine(self._x, gamma, self._c_x)
        self._y = self._game.domain(1).combine(self._y, gamma, self._c_y)

        # if we only needed two fixed point iters then we can be more aggressive
        if self._aggressive_stepsizes and fixed_point_iters < 3:
            self._gamma *= 1.2

    # the condition relies on the following quantity:
    # \delta = stepsize * (<Aw_y, old_w_x - new_w_x> +
    #                       <-A^Told_w_x, old_w_y - new_w_y>)
    #                     + V_old_x (new_x) + V_old_y (new_y),
    #
    # which must be nonpositive.
    def stepsize_condition_violated(self, stepsize, new_w_x, new_w_y, old_w_x,
                                    old_w_y, old_x, old_y):
        g_x = -self._game.utility_for(0, old_w_y)
        g_y = -self._game.utility_for(1, old_w_x)

        diff_x = self._game.domain(0).sequence_form(old_w_x) \
            - self._game.domain(0).sequence_form(new_w_x)
        diff_y = self._game.domain(1).sequence_form(old_w_y) \
            - self._game.domain(1).sequence_form(new_w_y)

        inner_prod = stepsize * (g_x.dot(diff_x) + g_y.dot(diff_y))
        assert not np.isnan(inner_prod)
        if inner_prod < 0:
            import ipdb
            ipdb.set_trace()

        divergence = self.bregman_divergence(new_w_x, new_w_y, old_x, old_y)
        assert not np.isnan(divergence)
        delta = inner_prod - divergence
        logging.debug("delta: {}, inner prod: {}, divergence: {}".format(
            delta, inner_prod, divergence))
        return delta > 0

    def bregman_divergence(self, x, y, x_center, y_center):
        return self._prox_weights[0] * self._game.domain(0).prox().bregman_divergence(x, x_center) \
            + self._prox_weights[1] * \
            self._game.domain(1).prox().bregman_divergence(y, y_center)


def mirror_prox_init(aggressive_stepsizes=False, **kwargs):
    def init(game):
        return MirrorProx(
            game, aggressive_stepsizes=aggressive_stepsizes, **kwargs)

    return init
