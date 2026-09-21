"""
will advance the code forward. regular output etc.
"""

import numpy as np
from collide import do_collision
from stream import do_stream


def advance_sim(grid, tau, max_step=1e3, VERBOSE=False, plot_int=100):
    """
    runs the sim stepping through collision and stream steps and eventually plotting
    some of the data.
    """

    for i in range(int(max_step)):
        if VERBOSE:
            print("Step: {i:0.0d}")

        # colllide
        if VERBOSE:
            print("Entering Collision step")
        do_collision(grid)

        # stream/prop 
        if VERBOSE:
            print("Entering Streaming step")
        do_stream(grid)

        # plot out data every so timesteps
        if int(max_step) % plot_int  == 0:
            plot_data(grid)

    