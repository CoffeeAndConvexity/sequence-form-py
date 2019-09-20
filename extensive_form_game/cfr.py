from matrix_game.simplex import SimplexDomain


class CounterfactualRegretMinimizer:
    def __init__(self, domain, initialize_regret_minimizer, name=None):
        self.domain = domain
        self.rms = [
            initialize_regret_minimizer(
                SimplexDomain(domain.information_set_num_sequences(info_set)))
            for info_set in range(domain.num_information_sets())
        ]
        self.strategy = domain.center()
        self.name = name

    def __call__(self, utility):
        for info_set in self.domain.infoset_traversal():
            begin = self.domain.information_set_first_sequence(info_set)
            end = self.domain.information_set_last_sequence(info_set)
            parent = self.domain.information_set_parent_sequence(info_set)
            ev = self.rms[info_set](utility[begin:end])
            if parent != self.domain.root_sequence():
                utility[parent] += ev
            self.strategy[begin:end] = self.rms[info_set].strategy

    def __str__(self):
        assert len(self.rms) > 0
        if self.name is None:
            return 'CFR(%s)' % self.rms[0]
        else:
            return self.name
