from __future__ import print_function

import logging
import argparse
import math
import sys
import time
import numpy as np

from extensive_form_game import blsp_reader
from extensive_form_game import libef_reader # this line can be commented in in order to get the libgg reader
from poker import kuhn
from poker import leduc
from poker import nlhe_river
from matrix_game import game as matrix_game
from matrix_game import regret as matrix_regret
from eqm import chambolle_pock as cp
from eqm import excessive_gap_technique as egt
from eqm import mirror_prox as mp
from eqm import regret as eqm_regret

algs = {
    'CP': lambda args: cp.ChambollePock,
    'EGT': lambda args: egt.excessive_gap_technique_init(
        aggressive_stepsizes=args.aggressive_stepsizes,
        init_gap=init_gap, init_update_x=init_update_x,
        allowed_eps_increase=allowed_eps_increase),
    'MP': lambda args: mp.mirror_prox_init(
        aggressive_stepsizes=args.aggressive_stepsizes),
    'HEDGE': lambda args: eqm_regret.regret_minimization_initializer(
        matrix_regret.hedge_initializer(
            1.0 / math.sqrt(num_iterations))),
    'RM': lambda args: eqm_regret.regret_minimization_initializer(
        matrix_regret.regret_matching_initializer()),
    'RM+': lambda args: eqm_regret.regret_minimization_initializer(
        matrix_regret.regret_matching_plus_initializer(),
        alternate=False, linear_averaging=False),
    'CFR+': lambda args: eqm_regret.regret_minimization_initializer(
        matrix_regret.regret_matching_plus_initializer(),
        alternate=True, linear_averaging=True, name='CFR+'),
    'CBA+': lambda args: eqm_regret.regret_minimization_initializer(
        matrix_regret.conic_blackwell_plus_initializer(),
        alternate=True, linear_averaging=True, name='CBA+'),
    'RM+_LINEAR': lambda args: eqm_regret.regret_minimization_initializer(
        matrix_regret.regret_matching_plus_initializer(),
        alternate=False, linear_averaging=True, name='CFR+'),
    'EGT_WARM': lambda args: egt.egt_warm_start_initializer(
        alg=eqm_regret.regret_minimization_initializer(
            matrix_regret.regret_matching_plus_initializer(),
            alternate=True, linear_averaging=True),
        aggressive_stepsizes=args.aggressive_stepsizes),
}

parser = argparse.ArgumentParser()
# Game params
parser.add_argument(
    '-g',
    '--game',
    default='kuhn',
    dest='game',
    help='Game to solve: kuhn, river, leduc, path to .blsp file, path to .capnp file')
parser.add_argument(
    '-r',
    '--num_ranks',
    type=int,
    default=3,
    help='Number of ranks in the deck. Only works for Leduc.')

# Algorithm params
parser.add_argument('-a', '--algorithm',
                    default=','.join(algs.keys()), dest='alg',
                    help='Available algorithms: %s' % ', '.join(algs.keys()) +
                    '. (a comma-separated list chooses several algorithms, ' +
                    'e.g. \'EGT,RM+\')')
parser.add_argument(
    '-t',
    '--num_iterations',
    type=int,
    default=100,
    help='number of algorithm iterations')
parser.add_argument(
    '--eps_threshold', type=float, default=-1.0, help='stopping threshold')
parser.add_argument(
    '-s',
    '--aggressive_stepsizes',
    action='store_true',
    default=False,
    dest='aggressive_stepsizes',
    help='use aggressive stepsizing in EGT and Mirror Prox')

# DGF params
parser.add_argument(
    '--prox_scalar',
    type=float,
    default=1,
    help='Scalar value applied to the whole prox function.\
                    Default uses 1.0')
parser.add_argument(
    '-w',
    '--prox_infoset_weights',
    default="all_one",
    dest='prox_infoset_weights',
    help='The weighting scheme used to construct entropy DGF. \
                    Option: all_one, kroer15, kroer17.')

# EGT-only params
parser.add_argument(
    '--init_gap',
    type=float,
    default=-1.0,
    help='find initial smoothing parameters that satisfy this gap')
parser.add_argument(
    '--allowed_eps_increase',
    type=float,
    default=-1.0,
    help='If > 0 then disallow iterations that cause a decrease in solution\
            quality by more than the given multiplicative factor')
parser.add_argument(
    '--init_update_x',
    action='store_true',
    default=False,
    dest='init_update_x',
    help='whether to update initial x during search for initial' +
    'smoothing parameters.')

# Args to do with output formatting
parser.add_argument(
    '--num_outputs', type=int, default=10, help='number of epsilon outputs')
parser.add_argument('--gnuplot', default='/dev/null', help='gnuplot output')
parser.add_argument(
    '-d',
    '--debug',
    action='store_true',
    default=False,
    dest='debug',
    help='display debug output')
parser.add_argument(
    '--csv',
    action='store_true',
    default=False,
    dest='to_csv',
    help='whether to output in CSV format')
parser.add_argument(
    '--pretty_print',
    action='store_true',
    default=True,
    dest='pretty_print',
    help='whether to output in CSV format')
parser.add_argument(
    '--log_scale',
    action='store_true',
    default=False,
    dest='log_scale',
    help='whether to output the num_outputs outputs spaced ' +
    'according to linear or log-scale x axes.')

args = parser.parse_args()

num_iterations = args.num_iterations
num_outputs = args.num_outputs
eps_threshold = args.eps_threshold
debug = args.debug
to_csv = args.to_csv
pretty_print = args.pretty_print
log_scale = args.log_scale
gnuplot_out = open(args.gnuplot, 'wt')
init_gap = args.init_gap
init_update_x = args.init_update_x
allowed_eps_increase = args.allowed_eps_increase
if to_csv:
    print('iters,gradients,eps,profile_val,algorithm,time')
elif debug:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)

if args.game == 'river':
    game = nlhe_river.init_efg(
        prox_infoset_weights=args.prox_infoset_weights,
        prox_scalar=args.prox_scalar)
elif args.game == 'river_big':
    game = nlhe_river.init_efg_big(
        prox_infoset_weights=args.prox_infoset_weights,
        prox_scalar=args.prox_scalar)
elif args.game == 'kuhn_matrix':
    game = kuhn.init_matrix()
elif args.game == 'kuhn':
    game = kuhn.init_efg(
        prox_infoset_weights=args.prox_infoset_weights,
        prox_scalar=args.prox_scalar)
elif args.game == 'leduc':
    game = leduc.init_efg(
        num_ranks=args.num_ranks,
        prox_infoset_weights=args.prox_infoset_weights,
        prox_scalar=args.prox_scalar)
elif '.blsp' in args.game:
    game = blsp_reader.make_efg_from_file(
        args.game,
        prox_infoset_weights=args.prox_infoset_weights,
        prox_scalar=args.prox_scalar)
elif '.game' in args.game: # this if condition can be commented in for the libgg reader
    game = libef_reader.make_efg_from_file(
        args.game,
        prox_infoset_weights=args.prox_infoset_weights,
        prox_scalar=args.prox_scalar)
else:
    assert False, 'unknown game %s' % args.game

algs_to_run = []

algs_arg = set(args.alg.upper().split(','))
for alg in algs_arg:
    if alg not in algs:
        print('Unknown algorithm "%s"' % alg)
        sys.exit(1)
    else:
        algs_to_run += [algs[alg](args)]

alg_names = []

if log_scale:
    # print_seq = np.logspace(
    #     1, np.log10(num_iterations), num_outputs, dtype=int)
    print_seq = np.unique(
        np.insert(
            np.geomspace(1, num_iterations, num_outputs, dtype=int), 0, 0))
else:
    print_seq = np.linspace(0, num_iterations, num_outputs, dtype=int)
for alg_idx, alg in enumerate(algs_to_run):
    t0 = time.time() # start timer
    opt = alg(game)
    total_time = time.time() - t0
    if not to_csv:
        print(opt)
        print('iters\tgrads\teps\t\tprofile_val\ttime')
    eps_initial = opt.epsilon()
    profile_val_initial = opt.profile_value()

    alg_names.append(str(opt))
    gradient_computations = opt.gradient_computations()

    t = 0
    print('$alg%d << EOD' % (alg_idx), file=gnuplot_out)
    print(0, 0, eps_initial, file=gnuplot_out)

    for i in range(len(print_seq)):
        # delta = num_iterations/num_outputs + (i < num_iterations % num_outputs)
        if i > 0:
            delta = print_seq[i] - print_seq[i - 1]
        else:
            delta = print_seq[0]

        t0 = time.time()
        opt.iterate(delta)
        total_time += time.time() - t0
        eps = opt.epsilon()
        profile_val = opt.profile_value()
        if to_csv:
            print('{iters},{gradients},{eps},{profile_val},{algorithm},{time}'.format(
                iters=print_seq[i],
                gradients=opt.gradient_computations(),
                eps=eps,
                profile_val=profile_val,
                algorithm=opt,
                time=total_time
            ))
        elif pretty_print:
            if False and str(opt) == 'ExcessiveGapTechnique':
                print(
                    '{iters}\t{grads}\t{eps:.6f}\t{profile_val:.6f}\t{egv:.6f}'.
                    format(
                        iters=print_seq[i],
                        grads=opt.gradient_computations(),
                        eps=eps,
                        profile_val=profile_val,
                        egv=opt.excessive_gap(), ))
            else:
                print('{iters}\t{grads}\t{eps:.6f}\t{profile_val:.6f}\t{time:.6f}'.format(
                    iters=print_seq[i],
                    grads=opt.gradient_computations(),
                    eps=eps,
                    profile_val=profile_val,
                    time=total_time
                ))
        else:
            print(print_seq[i], opt.gradient_computations(), eps, profile_val)
        print(t, opt.gradient_computations(), eps, total_time, file=gnuplot_out)
        if eps < eps_threshold:
            break

    print('EOD', file=gnuplot_out)

print("""
set terminal png
""", file=gnuplot_out)

print(
    """
set output 'by_iter.png'
set ylabel 'epsilon'
set xlabel '# iterations'
set logscale y
set logscale x
""",
    file=gnuplot_out)

for alg_idx, name in enumerate(alg_names):
    if alg_idx == 0:
        line = 'plot '
    else:
        line = ' '

    line += '"$alg%d" using 1:3 with lines lw 2 title "%s"' % (alg_idx, name)

    if alg_idx + 1 < len(alg_names):
        line += ',\\'

    print(line, file=gnuplot_out)

print(
    """
set output 'by_grad.png'
set xlabel '# gradients'
""",
    file=gnuplot_out)

for alg_idx, name in enumerate(alg_names):
    if alg_idx == 0:
        line = 'plot '
    else:
        line = ' '

    line += '"$alg%d" using 2:3 with lines lw 2 title "%s"' % (alg_idx, name)

    if alg_idx + 1 < len(alg_names):
        line += ',\\'

    print(line, file=gnuplot_out)

gnuplot_out.close()
