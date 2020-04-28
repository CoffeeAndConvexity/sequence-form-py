import sys
from collections import defaultdict
import numpy as np
import scipy.sparse as sparse
# import extensive_form_game as efg
from .extensive_form_game import ExtensiveFormGame
import capnp


def make_efg_from_file(filename, prox_infoset_weights=False, prox_scalar=-1):
    capnp.remove_import_hook()
    game_capnp = capnp.load("/Users/christiankroer/Dropbox/research/code/libefg/libefg/schema/game.capnp")
    game_obj = game_capnp.Game.read(open(filename,"rb"))
    # check that the first line says that it's a zero-sum game

    num_infosets_p1 = len(game_obj.treeplexPl1.infosets)
    num_infosets_p2 = len(game_obj.treeplexPl2.infosets)



    # get p1 treeplex
    first_p1 = list(map(lambda x: x.startSequenceId, game_obj.treeplexPl1.infosets))
    first_p2 = list(map(lambda x: x.startSequenceId, game_obj.treeplexPl2.infosets))
    end_p1 = list(map(lambda x: x.endSequenceId, game_obj.treeplexPl1.infosets))
    end_p2 = list(map(lambda x: x.endSequenceId, game_obj.treeplexPl2.infosets))
    parent_p1 = list(map(lambda x: x.parentSequenceId, game_obj.treeplexPl1.infosets))
    parent_p2 = list(map(lambda x: x.parentSequenceId, game_obj.treeplexPl2.infosets))

    num_sequences_p1 = max(parent_p1) + 1
    num_sequences_p2 = max(parent_p2) + 1

    # get payoff matrix
    p1_ind = list(map(lambda x: x.sequencePl1, game_obj.payoffMatrix.entries))
    p2_ind = list(map(lambda x: x.sequencePl2, game_obj.payoffMatrix.entries))
    payoff_values = list(map(lambda x: x.chanceFactor * x.payoffPl2, game_obj.payoffMatrix.entries))

    A = sparse.csr_matrix((payoff_values, (p1_ind, p2_ind)),
        shape=(num_sequences_p1, num_sequences_p2))

    return ExtensiveFormGame(
        "LIBGG EFG",
        A, (first_p1, first_p2), (end_p1, end_p2), (parent_p1, parent_p2),
        prox_infoset_weights=prox_infoset_weights,
        prox_scalar=prox_scalar)
