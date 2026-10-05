"""
will advance the code forward. regular output etc.
"""

import os
import numpy as np
from .collide import do_collision, calc_u, calc_rho
from .stream import do_stream
from .plotfile import plot_data


def advance_sim(grid, tau, dt_phy=1., max_step=1e3, VERBOSE=False, plot_int=100, plot_dir='./'):
    """
    runs the sim stepping through collision and stream steps and eventually plotting
    some of the data.
    """

    plot_data(grid, outname=os.path.join(plot_dir, 'plt_after_initialization.npy'))
    for i in range(int(max_step)):

        if VERBOSE:
            print(f"Step: {i:d}\t Time: {i*dt_phy:0.4e} s")

        # colllide
        if VERBOSE:
            print("Entering Collision step")
        do_collision(grid, tau)


        # stream/prop 
        if VERBOSE:
            print("Entering Streaming step")
            print(f"Max velx: {np.max(grid.uvec[:, :, 0])}")
            print(f"Max vely: {np.max(grid.uvec[:, :, 1])}")
        do_stream(grid)
        if VERBOSE:
            print("End stream")
            calc_rho(grid)
            calc_u(grid)
            print(f"Max velx: {np.max(grid.uvec[:, :, 0])}")
            print(f"Max vely: {np.max(grid.uvec[:, :, 1])}")

        # plot out data every so timesteps
        if i % plot_int  == 0:
            #print(f"Total Mass at step {i}: {np.sum(grid.rho):0.3e}")
            outfile = f"{plot_dir}/plt{i:07d}.npy"
            plot_data(grid, outname=outfile)


    