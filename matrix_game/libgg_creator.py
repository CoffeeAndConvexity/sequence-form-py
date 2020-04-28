import sys
from collections import defaultdict
import numpy as np
import scipy.sparse.linalg as linalg
import capnp

def make_capnp_from_matrix(A, filename):
    capnp.remove_import_hook()
    game_capnp = capnp.load("/Users/christiankroer/Documents/research/code/rust/libefg/libefg/schema/game.capnp")

    game = game_capnp.Game.new_message()
    game.treeplexPl1 = game_capnp.Treeplex.new_message()
    game.treeplexPl2 = game_capnp.Treeplex.new_message()
    game.payoffMatrix = game_capnp.PayoffMatrix.new_message()

    game.treeplexPl1.init('infosets', 1)
    game.treeplexPl2.init('infosets', 1)

    game.treeplexPl1.infosets[0].endSequenceId = A.shape[0] - 1
    game.treeplexPl1.infosets[0].parentSequenceId = A.shape[0]

    game.treeplexPl2.infosets[0].endSequenceId = A.shape[1] - 1
    game.treeplexPl2.infosets[0].parentSequenceId = A.shape[1]

    game.payoffMatrix.init('entries', A.shape[0] * A.shape[1])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            game.payoffMatrix.entries[i*A.shape[1] + j].sequencePl1 = i
            game.payoffMatrix.entries[i*A.shape[1] + j].sequencePl2 = j
            game.payoffMatrix.entries[i*A.shape[1] + j].payoffPl1 = -float(A[i,j])
            game.payoffMatrix.entries[i*A.shape[1] + j].payoffPl2 = float(A[i,j])
            game.payoffMatrix.entries[i*A.shape[1] + j].chanceFactor= 1.0

    game.payoffMatrix.l2norm = float(linalg.svds(A, k=1, return_singular_vectors=False)[0]) # largest singular value of A = matrix 2-norm of A = Lipschitz constant of the gradient
    game.payoffMatrix.maxnorm = float(A.max() - A.min())
    f = open(filename, 'w+b')
    game.write(f)

