"""
compute the collision step w/ a BGK operator
"""

def calc_rho(grid):
    """
    calc rho at every point on the grid
    """

    self.rho = np.sum(grid.grid, axis=2)

def calc_u(grid):
    """
    calc u at every point on the grid
    """

    # construct vector for u

    # should prolly unroll this. doing lots of unncessary multiplying by zero
    self.u_vec[:, :, 0] = np.sum(grid.grid, axis=2, weights=cx) / grid.rho
    self.u_vec[:, :, 1] = np.sum(grid.grid, axis=2, weights=cy) / grid.rho

def calc_feq(grid):
    """
    calculate the equilibrium distribution func 
    for each node

    takes in grid

    returns feq on each node
    """

    # WIP
    cs2_inv = 3.
    cs4_2_inv = 2.*9.

    vel_term = 1. + cs2_inv * (grid.cx * grid.u_vec[:, :, 0] + 
                               grid.cy * grid.u_vec[:, :, 1]) 
                  + cs4_2_inv * (grid.u_vec[:, :, 0]**2 * (grid.cx**2 - 1./cs2_inv) + 
                               grid.u_vec[:, :, 1]**2 * (grid.cy**2 - 1./cs2_inv) +
                               2.*grid.u_vec[:, :, 0]*grid.u_vec[:, :, 1] * (grid.cx*grid.cy))
    grid.grid_eq = weights * grid.rho * vel_term


def do_collision(grid, tau, dt):
    """
    calculates the collision on the grid and 
    updates the grid.

    Takes in grid (type Grid) and tau (relaxation time) and timestep, dt, I guess
    """

    calc_rho(grid)
    calc_u(grid)

    calc_freq(grid)
    omega = dt/tau

    grid.grid = (1-omega) * grid.grid + omega * grid.grid_eq

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

    grid.grid = new_grid
            