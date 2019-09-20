from itertools import chain
from itertools import product
import numpy as np
from scipy.sparse import csr_matrix
from extensive_form_game import extensive_form_game as efg
from .holdem_hands import compute_winner

def init_efg_big(prox_infoset_weights=False, prox_scalar=1):
    return init_efg(pot_size=100,
                    stacks=[100, 100],
                    hands=[
                        [
                            [[14, 14], [13, 13], [12, 12]],
                            [[0, 1], [0, 1], [0, 1]]
                        ],
                        [
                            [[14, 14], [13, 13], [12, 12]],
                            [[0, 1], [0, 1], [0, 1]]
                        ]
                    ],
                    min_raise=2,
                    pot_fractions=[0.33, 0.5, 0.66, 1],
                    board=np.array([[2, 7, 4, 5, 6], [2, 3, 0, 0, 1]]),
                    prox_infoset_weights=prox_infoset_weights,
                    prox_scalar=prox_scalar)
"""
hands should be a length 2 array with the range for each player in the format
hands[player][0][i] = i'th hand (e.g. [14, 14]),
hands[player][1][i] = i'th suit (e.g. [0, 1])

probabilites should give probabilites for each hand. If probabilities is None,
the hands are assumed to have uniform probability.

pot_fractions is the set of raise sizes as a percentage of current pot size.

cutoff is used in order to ignore fractions of pot size that are above cutoff * all-in

The default values create a Kuhn game, since the board is irrelevant and the
ranges are AA, KK, QQ.
"""
def init_efg(pot_size=2,
             stacks=[1, 1],
             hands=[
                 [
                     [[14, 14], [13, 13], [12, 12]],
                     [[0, 1], [0, 1], [0, 1]]
                 ],
                 [
                     [[14, 14], [13, 13], [12, 12]],
                     [[0, 1], [0, 1], [0, 1]]
                 ]
             ],
             min_raise=1,
             pot_fractions=[0.33, 0.5, 0.66, 1],
             board=np.array([[2, 7, 4, 5, 6], [2, 3, 0, 0, 1]]),
             prox_infoset_weights=False, prox_scalar=1):
    game_creator = GameState(pot_size, stacks, hands, board, min_raise,
                             pot_fractions)
    A = _dict_to_csr(game_creator.payoff)

    def make_numpy_array(d):
        a = np.zeros(len(d), int)
        for key, val in d.iteritems():
            a[key] = val
        return a

    first_p1 = make_numpy_array(game_creator.first[0])
    end_p1 = make_numpy_array(game_creator.end[0])
    parent_p1 = make_numpy_array(game_creator.parent[0])

    first_p2 = make_numpy_array(game_creator.first[1])
    end_p2 = make_numpy_array(game_creator.end[1])
    parent_p2 = make_numpy_array(game_creator.parent[1])

    return efg.ExtensiveFormGame("River %s EFG" % hands, A, (first_p1, first_p2),
                             (end_p1, end_p2), (parent_p1, parent_p2), game_creator.seq_to_str,
                             prox_infoset_weights=prox_infoset_weights,
                             prox_scalar=prox_scalar)

class GameState:
    def __init__(self, pot_size, stacks, hands, board, min_raise, pot_fractions):
        self.pot_size      = pot_size
        self.initial_pot   = pot_size
        self.stacks        = stacks
        self.hands         = hands
        self.board         = board
        self.min_raise     = min_raise
        self.pot_fractions = pot_fractions
        self.raise_allowed = True
        self.player        = 0
        self.actions       = []
        self.infosets      = [{}, {}]
        self.payoff        = {}
        self.num_sequences = [1, 1]
        self.history       = []
        self.seq_to_str    = [{}, {}]

        self.first  = [{}, {}]
        self.end   = [{}, {}]
        self.parent = [{}, {}]
        self.traverse_hands()

    def infoset(self, previous_seq, last_raise):
        if self.sequence(self.player) not in self.infosets[self.player]:
            num_infosets = len(self.infosets[self.player])
            num_actions  = self.num_actions(self.num_sequences[self.player], last_raise)
            self.infosets[self.player][self.sequence(self.player)] = num_infosets
            self.first[self.player][num_infosets] = self.num_sequences[self.player]
            self.end[self.player][num_infosets] = self.num_sequences[self.player] + num_actions
            self.parent[self.player][num_infosets] = previous_seq
            self.num_sequences[self.player] += num_actions
            assert self.parent[self.player][num_infosets] < self.first[self.player][num_infosets]
        return self.infosets[self.player][self.sequence(self.player)]

    def num_actions(self, first, last_raise):
        count = 0  # can always either check or call
        for pot_fraction in self.pot_fractions:
            size = int(pot_fraction * self.pot_size)
            if self.raise_allowed and \
               size >= last_raise and size >= self.min_raise and size < min(self.stacks[0], self.stacks[1]):
                self.seq_to_str[self.player][first + count] = self.sequence(self.player) + 'r' + str(pot_fraction)
                count += 1
        if self.raise_allowed:
            self.seq_to_str[self.player][first + count] = self.sequence(self.player) + 'a'
            count += 1  # all-in
        self.seq_to_str[self.player][first + count] = self.sequence(self.player) + 'c'
        count += 1 #  call
        if len(self.actions) > 0 and \
           (self.actions[-1] == 'a' or self.actions[-1][0] == 'r'):
            self.seq_to_str[self.player][first + count] = self.sequence(self.player) + 'f'
            count += 1  # fold
        return count

    def add_sequence(self, sequence):
        self.player = 1 - self.player
        self.actions.append(sequence)

    def remove_sequence(self):
        self.player = 1 - self.player
        self.actions.pop()

    def amount_won(self):
        # amount = self.initial_pot / 2
        # raises_won = self.pot_size - self.initial_pot
        # if self.actions[-1] == 'f':
        #     raises_won -= self.last_raise
        # amount += raises_won / 2  # the player made half the raises themself
        amount = self.pot_size / 2.0
        return self.prob * amount

    def sequence(self, player):
        return '{hand}/{seq}'.format(
            hand=self.hand[player],
            seq='/'.join(self.actions),
        )

    def game_history(self):
        return '{hand1}/{hand2}/{seq}'.format(
            hand1=self.hand[0],
            hand2=self.hand[1],
            seq='/'.join(self.actions),
        )

    def blocking(self, p1_idx, p2_idx):
        return (self.hands[0][0][p1_idx][0] == self.hands[1][0][p2_idx][0] and
                self.hands[0][1][p1_idx][0] == self.hands[1][1][p2_idx][0]) or \
                (self.hands[0][0][p1_idx][1] == self.hands[1][0][p2_idx][1] and
                 self.hands[0][1][p1_idx][1] == self.hands[1][1][p2_idx][1])

    def traverse_hands(self):
        for idx1 in range(len(self.hands[0][0])):
            non_blocking = filter(lambda idx2: not self.blocking(idx1, idx2),
                                  range(len(self.hands[1][0])))
            self.prob = 1.0 / (len(self.hands[0][0]) * len(non_blocking))
            for idx2 in non_blocking:
                self.hand = [
                    [self.hands[0][0][idx1], self.hands[0][1][idx1]],
                    [self.hands[1][0][idx2], self.hands[1][1][idx2]],
                ]
                self.previous_sequence_id = [-1, -1]
                self.showdown_winner = compute_winner(self.hand, self.board)
                self.traverse_betting(-1, -1, last_raise=0)

    def traverse_betting(self, parent_seq1, parent_seq2, last_raise):
        min_stack = min(self.stacks[0], self.stacks[1])
        player = self.player
        if player == 0:
            infoset = self.infoset(parent_seq1, last_raise)
        else:
            infoset = self.infoset(parent_seq2, last_raise)
        current_seq = self.first[player][infoset]
        new_seq1, new_seq2 = parent_seq1, parent_seq2
        assert self.previous_sequence_id[player] < current_seq

        for pot_fraction in self.pot_fractions:
            size = int(pot_fraction * self.pot_size)
            if self.raise_allowed and size >= self.min_raise and \
               size >= last_raise and size < min_stack:
                self.pot_size += size
                self.stacks[player] -= size
                self.add_sequence('r{0}'.format(pot_fraction))
                if player == 0:
                    new_seq1 = current_seq
                else:
                    new_seq2 = current_seq
                current_seq += 1
                self.traverse_betting(new_seq1, new_seq2, size)
                self.pot_size -= size
                self.stacks[player] += size
                self.remove_sequence()

        if self.raise_allowed:
            self.add_sequence('a')
            self.raise_allowed = False
            if player == 0:
                new_seq1 = current_seq
            else:
                new_seq2 = current_seq
            self.pot_size += self.stacks[player]
            current_seq += 1
            self.traverse_betting(new_seq1, new_seq2, self.stacks[player])
            self.pot_size -= self.stacks[player]
            self.raise_allowed = True
            self.remove_sequence()

        self.add_sequence('c')
        if player == 0:
            new_seq1 = current_seq
        else:
            new_seq2 = current_seq
        current_seq += 1
        if len(self.actions) == 1:
            self.traverse_betting(new_seq1, new_seq2, 0)
        else:
            self.pot_size += last_raise
            self.handle_leaf(new_seq1, new_seq2)
        self.pot_size -= last_raise
        self.remove_sequence()

        if len(self.actions) > 0 and self.actions[-1] != 'c':
            self.add_sequence('f')
            if player == 0:
                new_seq1 = current_seq
            else:
                new_seq2 = current_seq
            current_seq += 1
            self.pot_size -= last_raise
            self.handle_leaf(new_seq1, new_seq2)
            self.pot_size += last_raise
            self.remove_sequence()

    def handle_leaf(self, seq1, seq2):
        assert seq1 >= 0
        assert seq2 >= 0
        if self.showdown_winner == 1 or (len(self.actions) % 2 == 0 and self.actions[-1] == 'f'):
            self.payoff[seq1, seq2] = -self.amount_won() #  p1 is minimizing
        elif self.showdown_winner == -1 or (len(self.actions) % 2 == 1 and self.actions[-1] == 'f'):
            self.payoff[seq1, seq2] = self.amount_won()


def _dict_to_csr(term_dict):
    term_dict_v = list(term_dict.itervalues())
    term_dict_k = list(term_dict.iterkeys())
    shape = list(np.repeat(np.asarray(term_dict_k).max() + 1, 2))
    csr = csr_matrix((term_dict_v, zip(*term_dict_k)), shape=shape)
    return csr
