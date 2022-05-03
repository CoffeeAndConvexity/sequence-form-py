import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar


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


class ConicBlackwellPlus:
    def __init__(self, dimension):
        self._dimension = dimension
        self.strategy = np.ones(dimension) / dimension
        self.regret_hat = np.zeros(dimension)
        self.regret_tilde = 0.0

    # solving for y_tilde using trick
    def solve_for_y_tilde_1(self,u_tilde,u_hat):
        def fn(x):
            loc_vec = u_hat+x*np.ones(len(u_hat))
            return x + np.sum(np.where(loc_vec>0,loc_vec,0))-u_tilde

        sol=fsolve(fn,u_tilde)
        sol=sol[0]
        return sol

    # solving for y_tilde using binary search
    def solve_for_y_tilde_2(self,u_tilde,u_hat):
        def fn(z):
            loc_vec = u_hat+z*np.ones(len(u_hat))
            loc_vec=np.where(loc_vec>0,loc_vec,0)
            res=((z-u_tilde)**2) + ((np.linalg.norm(loc_vec))**2)
            return res

        sol=minimize_scalar(fn,method='Brent',tol=0.0000001)
        sol=sol.x
        return sol

    # compute the projection onto the cone
    def projection_on_cone(self,u_loc_tilde,u_loc_hat):
        # computing optimal y_tilde
        y_tilde_star = self.solve_for_y_tilde_1(u_loc_tilde,u_loc_hat)
        # proj_u_tilde = u_tilde - y_tilde
        proj_u_tilde = u_loc_tilde - y_tilde_star
        # proj_u_hat = (u_hat+y_tilde*e)^+
        u_loc=u_loc_hat+y_tilde_star*np.ones(len(u_loc_hat))
        proj_u_hat =np.where(u_loc>0,u_loc,0)
        return proj_u_hat,proj_u_tilde

    def __call__(self, utility):
        value = np.dot(self.strategy, utility)

        # intermediate value for u_tilde and u_hat
        u_loc_tilde = self.regret_tilde - value
        u_loc_hat = self.regret_hat + utility

        # projection of (u_loc_tilde,u_loc_hat) onto cone C
        proj_u_hat,proj_u_tilde = self.projection_on_cone(u_loc_tilde,u_loc_hat)

        # update self.regret_hat, self.regret_tilde
        self.regret_tilde = proj_u_tilde
        self.regret_hat = proj_u_hat

        # update strategy
        if sum(proj_u_hat)==0:
            self.strategy=np.ones(self._dimension)/self._dimension
            # print('degenerate proj')
        else:
            self.strategy = proj_u_hat/sum(proj_u_hat)
            # print('non-degenerate proj')

        return value

    def __str__(self):
        return 'ConicBlackwell+'


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

def conic_blackwell_plus_initializer():
    def init(domain):
        return ConicBlackwellPlus(domain.dimension())

    return init
