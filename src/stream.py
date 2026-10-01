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
    apply_bc(grid, new_grid)
    # save new grid in current grid
    grid.grid = new_grid

def apply_bc(grid, new_grid):

    for i, bc_wall in enumerate((grid.bc_vertical, grid.bc_horizontal)):
        for side, bc_type in zip((0, -1), bc_wall):
            if bc_type == 'periodic':
                periodic_bc(grid, new_grid, side, i)
            elif bc_type == 'bounceback':
                bounceback_bc(grid, new_grid, side, i, grid.bc_vertical_kwarg, grid.bc_horizontal_kwarg)
            else:
                raise RuntimeError(f"Boundary Condition: {bc_type} not implemented")

def periodic_bc(grid, new_grid, direction, wall):
    # just verify this worked as intended
     # check periodic. currently no check on diagonal entries
    try:
        # vertical wall
        if wall == 0:
            # left side
            if direction == 0:
                assert(np.all(np.abs(new_grid[0, :, 1] - grid.grid[-1, :, 1]) < 1e-8))
            # right side
            elif direction == -1:
                assert(np.all(np.abs(new_grid[-1, :, 3] - grid.grid[0, :, 3]) < 1e-8))

        # horizontal wall
        elif wall == 1:
            # top
            if direction == 0:
                assert(np.all(np.abs(new_grid[:, 0, 2] - grid.grid[:, -1, 2]) < 1e-8))
            # bottom
            elif direction == -1:
                assert(np.all(np.abs(new_grid[:, -1, 4] - grid.grid[:, 0, 4]) < 1e-8))

    except AssertionError:
        print("The following should be equal:")
        print(grid.grid[-1, 0 , 1])
        print(new_grid[0, 0, 1])
        print(grid.grid[0, -1 , 2])
        print(new_grid[0, 0, 2])
        raise RuntimeError('Something went wrong in periodic BCs')

   
def wall_contrib(weight, rho_w, c_i, U_w):
    cs2_inv = 3.
    cx, cy = c_i
    U_wx, U_wy = U_w
    return -2.*cs2_inv*weight*rho_w*(cx * U_wx + cy * U_wy)

def bounceback_bc(grid, new_grid, direction, wall, bc_vertical_kwarg, bc_horizontal_kwarg):
    # set density at wall
    rho_w = np.mean(grid.rho)

    # bounce off the wall one step
    # vertical wall
    if wall == 0:
        #setup moving wall 
        if bc_vertical_kwarg is not None:
            Uwall_left, Uwall_right = bc_vertical_kwarg['Uwall']
        else:
            Uwall_left = [0., 0.]
            Uwall_right = [0., 0.]
        # left side
        if direction == 0:
            # bounce all on left
            for l_i, r_i in grid.vertical_wall:
                wall_term = wall_contrib(grid.weights[l_i], rho_w,
                                         (grid.cx[l_i], grid.cy[l_i]), Uwall_left) 
                new_grid[:, 0, r_i] = grid.grid[:, 0, l_i] + wall_term

        # right side
        elif direction == -1:
            # bounce all on right
            for l_i, r_i in grid.vertical_wall:
                wall_term = wall_contrib(grid.weights[r_i], rho_w,
                                         (grid.cx[r_i], grid.cy[r_i]), Uwall_right)
                new_grid[:, -1, l_i] = grid.grid[:, -1, r_i] + wall_term

    # horizontal wall
    elif wall == 1:
        #setup moving wall 
        if bc_horizontal_kwarg is not None:
            Uwall_bot, Uwall_top = bc_horizontal_kwarg['Uwall']
        else:
            Uwall_bot = [0., 0.]
            Uwall_top = [0., 0.]
        # bot
        if direction == 0:
            # bounce on top
            for d_i, u_i in grid.vertical_wall:
                wall_term = wall_contrib(grid.weights[d_i], rho_w,
                                         (grid.cx[d_i], grid.cy[d_i]), Uwall_bot) 
                new_grid[0, :, u_i] = grid.grid[0, :, d_i] + wall_term
        # top
        elif direction == -1:
            #bounce bot wall
            for d_i, u_i in grid.vertical_wall:
                wall_term = wall_contrib(grid.weights[u_i], rho_w,
                                         (grid.cx[u_i], grid.cy[u_i]), Uwall_top) 
                new_grid[-1, :, d_i] = grid.grid[-1, :, u_i] + wall_term

