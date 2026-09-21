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

    grid