from __future__ import print_function
import sys
from collections import defaultdict
import numpy as np
import scipy.sparse as sparse
import extensive_form_game as efg


def make_efg_from_file(filename, prox_infoset_weights=False, prox_scalar=-1):
    with open(filename, 'r') as blsp_file:
        # check that the first line says that it's a zero-sum game
        assert blsp_file.readline() == 'CONSTANT_SUM 0\n'

        num_infosets_p1, num_sequences_p1, num_payoffs_p1, \
            num_infosets_p2, num_sequences_p2, num_payoffs_p2 = map(
                int, blsp_file.readline().strip().split())

        # get p1 treeplex
        first_p1 = map(int, blsp_file.readline().strip().split())
        end_p1 = map(lambda x: int(x) + 1,
                     blsp_file.readline().strip().split())
        parent_p1 = map(int, blsp_file.readline().strip().split())

        # get p2 treeplex
        first_p2 = map(int, blsp_file.readline().strip().split())
        end_p2 = map(lambda x: int(x) + 1,
                     blsp_file.readline().strip().split())
        parent_p2 = map(int, blsp_file.readline().strip().split())

        # get payoff matrix
        payoff_indptr = map(int, blsp_file.readline().strip().split())
        payoff_indices = map(int, blsp_file.readline().strip().split())
        payoff_values = map(lambda x: -float(x),
                            blsp_file.readline().strip().split())

        A = sparse.csr_matrix(
            (payoff_values, payoff_indices, payoff_indptr),
            shape=(num_sequences_p1, num_sequences_p2))

    return efg.ExtensiveFormGame(
        "BLSP EFG",
        A, (first_p1, first_p2), (end_p1, end_p2), (parent_p1, parent_p2),
        prox_infoset_weights=prox_infoset_weights,
        prox_scalar=prox_scalar)
