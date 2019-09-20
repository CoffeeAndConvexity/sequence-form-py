from collections import defaultdict
import numpy as np
import gurobipy as grb


class SequenceFormProgram:
    def __init__(self, game, player, opponent):
        self.game = game
        self.player = player
        self.opponent = opponent
        opponent_num_seqs = self.game.domain(self.opponent).dimension()
        self.child_infosets = defaultdict(list)
        self.seq_to_infoset = np.zeros(opponent_num_seqs)
        self.seq_to_infoset[0] = 0
        for i in range(self.game.domain(self.opponent).num_information_sets()):
            begin  = self.game.domain(self.opponent)._begin[i]
            end    = self.game.domain(self.opponent)._end[i]
            parent = self.game.domain(self.opponent)._parent[i]
            self.child_infosets[parent].append(i+1)
            self.seq_to_infoset[begin: end] = i+1

        self.model = grb.Model("Limited lookahead")
        self.add_variables()
        self.add_objective()
        self.add_constraints()
        self.model.optimize()
        self.model.printStats()

    def add_variables(self):
        # indexed as [infosetid]. Note that we expect information sets
        # to be 1-indexed, but the code corrects for when this is not the case
        # indexed as [infosetid][action.name]
        self.strategy_vars_infoset  = {}
        self.strategy_vars = self.model.addVars(
            self.game.domain(self.player).dimension(), ub=1.0,
            # name=["x{0}".format(i)
            #       for i in range(self.game.domain(self.player).dimension())]
        )
        self.dual_vars = self.model.addVars(
            self.game.domain(self.opponent).num_information_sets()+1,
            lb=-grb.GRB.INFINITY,
            # name=["v{0}".format(i)
            #       for i in range(self.game.domain(self.player).num_information_sets())]
        )
        self.model.update()

    def add_objective(self):
        self.model.setObjective(self.dual_vars[0], grb.GRB.MINIMIZE)

    def add_constraints(self):
        # sequence-form constraint for player
        self.model.addConstr(self.strategy_vars[0] == 1)
        for i in range(self.game.domain(self.player).num_information_sets()):
            begin  = self.game.domain(self.player)._begin[i]
            end    = self.game.domain(self.player)._end[i]
            parent = self.game.domain(self.player)._parent[i]
            sequence_vars = [self.strategy_vars[i] for i in range(begin, end)]
            parent_var = self.strategy_vars[parent]
            self.model.addConstr(grb.quicksum(sequence_vars) == parent_var)

        # payoff constraint for opponent (whose dual we are solving)
        for i in range(self.game.domain(self.opponent).dimension()):
            lhs = 1.0 * self.dual_vars[self.seq_to_infoset[i]]
            for dual in self.child_infosets[i]:
                lhs -= self.dual_vars[dual]

            if self.player == 0:
                A = self.game._A_T
            else:
                A = -self.game._A
            rhs = grb.LinExpr()
            for j in A[i].nonzero()[1]:
                if i == 0:
                    print j, A[i,j]
                rhs += A[i,j] * self.strategy_vars[j]

            self.model.addConstr(lhs >= rhs)

