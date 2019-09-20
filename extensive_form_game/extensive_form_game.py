from __future__ import print_function
import sys
from collections import defaultdict
import numpy as np
from scipy.sparse import isspmatrix_lil, isspmatrix_csr
from .treeplex import TreeplexDomain


class ExtensiveFormGame:
    """
    represents the saddle-point problem:
    min_{x\in\Delta} max_{y\in\Delta} x'Ay

    i.e., x, the first player, is the minimizer

    Expects A and reach to be of type scipy.sparse.lil_matrix or
    scipy.sparse.csr_matrix
    """

    def __init__(self,
                 name,
                 A,
                 first,
                 end,
                 parent,
                 seq_to_str=None,
                 prox_infoset_weights=False,
                 prox_scalar=1,
                 reach=None,
                 all_negative=False,
                 offset=0,
                 B=None):
        if seq_to_str is None:
            seq_to_str = [defaultdict(), defaultdict()]
        self._name = name
        assert isspmatrix_lil(A) or isspmatrix_csr(A)
        if isspmatrix_csr(A):
            self._A = A
        else:
            self._A = A.tocsr()
        if reach is not None:
            if isspmatrix_csr(reach[0]):
                self._reach = reach
            else:
                self._reach = (reach[0].tocsr(), reach[1].tocsr())
        self._A_T = self._A.transpose()
        # print(A, first, end, parent)
        self._domains = (TreeplexDomain(
            self._A.get_shape()[0],
            first[0],
            end[0],
            parent[0],
            seq_to_str[0],
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar), TreeplexDomain(
                self._A.get_shape()[1],
                first[1],
                end[1],
                parent[1],
                seq_to_str[1],
                prox_infoset_weights=prox_infoset_weights,
                prox_scalar=prox_scalar))
        self.all_negative = all_negative
        self.offset = offset
        if B is not None:
            if isspmatrix_csr(B):
                self._B = B
            else:
                self._B = B.tocsr()
        else:
            self._B = None

    def domain(self, player):
        return self._domains[player]

    def profile_epsilon(self, x, y):
        value = self.profile_value(x, y)
        br_x, _ = self.domain(0).support(self.utility_for(0, y))
        br_y, _ = self.domain(1).support(self.utility_for(1, x))
        return br_x + br_y, br_x - value, br_y + value, value

    def profile_value(self, x, y):
        seq = self.domain(0).sequence_form(x)
        return np.dot(seq, self.utility_for(0, y))

    def max_infoset_regret(self, x, y):
        return max(
            self.max_player_infoset_regret(0, x, y),
            self.max_player_infoset_regret(1, y, x))

    def max_player_infoset_regret(self, player, strategy, opponent_strategy):
        g = self.utility_for(player, opponent_strategy)
        regrets = self.domain(player).infoset_regrets(g, strategy)
        return np.max(regrets * self.reach(player, opponent_strategy))

    def sum_of_player_infoset_regret(self, player, strategy,
                                     opponent_strategy):
        g = self.utility_for(player, opponent_strategy)
        regrets = self.domain(player).infoset_regrets(g, strategy)
        return np.sum(regrets * self.reach(player, opponent_strategy))

    def utility_for(self, player, opponent_strategy):
        seq = self.domain(1 - player).sequence_form(opponent_strategy)
        if player == 0:
            return -self._A.dot(seq)

        assert player == 1
        return self._A_T.dot(seq)

    def payoff_max_norm(self):
        return max(self._A.max(), -self._A.min())

    def reach(self, player, opponent_strategy):
        if self._reach is None:
            raise ValueError(
                "The reach argument in __init__ has to be set in order to call reach()"
            )
        seq = self.domain(1 - player).sequence_form(opponent_strategy)
        return self._reach[player].dot(seq)

    def print_payoff_matrix(self,
                            f=sys.stdout,
                            negate=False,
                            all_negative=False):
        constant_term = 0
        if all_negative:
            constant_term = np.max(np.absolute(self._A)) + 1
        if negate:
            if self._B is None:
                B = -np.copy(self._A)
            else:
                B = self._B
        else:
            B = np.transpose(self._A).copy()
        if self._B is None:
            B.data -= constant_term
        for row in B.todense():
            np.savetxt(f, row, fmt="%i")

    def print_matrices(self, f=sys.stdout, all_negative=False):
        if all_negative:
            print("CS", self.offset, file=f)
        else:
            print("GS", file=f)
        print(self.domain(0).dimension(), self.domain(1).dimension(), file=f)
        self.print_payoff_matrix(f=f, negate=True, all_negative=all_negative)
        self.print_payoff_matrix(f=f, negate=False, all_negative=all_negative)
        self.domain(0).print_sequence_form_constraints(f=f)
        self.domain(1).print_sequence_form_constraints(f=f)

    def __str__(self):
        return 'ExtensiveFormGame(%s, %dx%d)' % (self._name, self._A.shape[0],
                                                 self._A.shape[1])
