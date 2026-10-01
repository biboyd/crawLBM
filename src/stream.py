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

    # using numpy's roll does periodic
    for idx in range(grid.nf):
        new_grid[:, :, idx] = np.roll(
                np.roll(grid.grid[:, :, idx], axis=0, shift=grid.cx[idx]),
                        axis=1, shift=grid.cy[idx])

    # apply BCs
    apply_bc(grid, new_grid, bc_type=grid.bc)
    # save new grid in current grid
    grid.grid = new_grid


def apply_bc(grid, new_grid, bc_type='periodic' ):
    if bc_type == 'periodic':
        periodic_bc(grid, new_grid)
    elif bc_type == 'bounceback':
        bounceback_bc(grid, new_grid)
    else:
        raise RuntimeError(f"Boundary Condition: {bc_type} not implemented")

def periodic_bc(grid, new_grid):
    # just verify this worked as intended
     # check periodic
    # x left
    try:
        assert(np.all(np.abs(new_grid[0, :, 1] - grid.grid[-1, :, 1]) < 1e-8))
        assert(np.all(np.abs(new_grid[:, 0, 2] - grid.grid[:, -1, 2]) < 1e-8))
        assert(np.all(np.abs(new_grid[-1, -1, 7] - grid.grid[0, 0, 7]) < 1e-8))
    except AssertionError:
         print(grid.grid[0, 0 , 7])
         print(new_grid[-1, 0, 7])
         raise RuntimeError('Something went wrong in periodic BCs')

   

def bounceback_bc(grid, new_grid):
    # bounce off the wall one step

    # left/right wall
    for l_i, r_i in grid.vertical_wall:
        #left wall
        new_grid[:, 0, r_i] = grid.grid[:, 0, l_i]
        #right wall
        new_grid[:, -1, l_i] = grid.grid[:, -1, r_i]

    # top/bot wall
    for t_i, b_i in grid.vertical_wall:
        #left wall
        new_grid[0, :, b_i] = grid.grid[0, :, t_i]
        #right wall
        new_grid[-1, :, t_i] = grid.grid[-1, :, b_i]
