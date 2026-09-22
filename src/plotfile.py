"""
will calc macro variables ie density 
and output them in a file.

will start by just outputting rho, vel in numpy array
"""

from collide import calc_rho, calc_u
import numpy as np

def plot_data(grid, outname='pfile'):

    # make sure u and rho are calculated
    calc_rho(grid)
    calc_u(grid)

    # use eos to calc pressure
    cs2_inv = 3.
    p = grid.rho/cs2_inv

    out_arr = np.dstack((grid.rho, p, grid.uvec))

    np.save(outname, out_arr)