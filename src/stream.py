"""
Will be used to compute the streaming 
step and move stuff around
"""

import numpy as np

def do_stream(grid):
    """
    copies grid and does a stream update
    """

    new_grid = np.empty_like(grid.grid)

    # copy index zero
    new_grid[:, :, 0] = grid.grid[:, :, 0]

    # gonna try using numpy's roll at least for cardinal directions

    # handle cardinal directions
    for idx in grid.cardinal:
        if (grid.cx[idx] != 0):
            new_grid[:, :, idx] = np.roll(grid.grid[:, :, idx], axis=0, shift=grid.cx[idx])
        elif (grid.cy[idx] != 0):
            new_grid[:, :, idx] = np.roll(grid.grid[:, :, idx], axis=1, shift=grid.cy[idx])

    #handle diagonal directions (can I just do two rolls? ya?)
    # just roll twice
    for idx in grid.diag:
            new_grid[:, :, idx] = np.roll(
                    np.roll(grid.grid[:, :, idx], axis=0, shift=grid.cx[idx]),
                                  axis=1, shift=grid.cy[idx])

    # hardcoded in periodic BCs but otherwise would do that step here

    grid.grid = new_grid