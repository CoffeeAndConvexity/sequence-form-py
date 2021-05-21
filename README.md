
# Prerequisites

- capnp and pycapnp: I recommend installing capnp first (see (here)[https://capnproto.org/install.html]), and then (pycapnp)[https://github.com/capnproto/pycapnp]
- python 3.7+

================

Example of how to run the code:

`~/anaconda2/bin/python driver.py -a egt,cfr+ -t 1000 -w kroer17 --num_output 21 --prox_scalar 1.0 --init_gap 0.001 --allowed_eps_increase 1.1 -g leduc -r 3`

This computes a  solution using EGT with aggressive stepsizing and CFR+

Here's an example of how to solve a game in the .game capcnp format:

`python driver.py -a cfr+ -t 1000 --num_output 10 -g ~/Documents/data/efg/games/leduc_2pl_3ranks.game`

================

authors on initial version of this code:
 - Christian Kroer (christian.kroer@columbia.edu)
 - Kevin Waugh (waugh@cs.cmu.edu)

other contributors:
- Gabriele Farina