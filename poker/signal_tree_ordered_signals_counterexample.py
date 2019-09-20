import numpy as np
from scipy.sparse import lil_matrix
from extensive_form_game import extensive_form_game as efg


DECK = ['J1', 'J2', 'K1', 'K2']

num_preflop_infosets = len(DECK)
num_postflop_infosets = len(DECK) * (len(DECK)-1)
num_preflop_sequences = 2 * num_preflop_infosets


first_p1 = np.array()


game = efg.ExtensiveFormGame(
    "Signal_tree_ordered_signals_counterexample%d EFG" % num_ranks,
    A,
    (first_p1, first_p2),
    (end_p1, end_p2),
    (parent_p1, parent_p2),
    prox_infoset_weights=prox_infoset_weights,
    prox_scalar=prox_scalar,
    reach=(reach[0].tocsr(), reach[1].tocsr()),
    all_negative=all_negative, )
