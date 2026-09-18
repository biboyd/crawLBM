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

    self.u_vec[:, :, 0] = np.sum(grid.grid, axis=2, weights=cy) / grid.rho
    self.u_vec[:, :, 1] = np.sum(grid.grid, axis=2, weights=cy) / grid.rho

def calc_feq(grid):
    """
    calculate the equilibrium distribution func 
    for each node

    takes in grid

    returns feq on each node
    """

    # WIP
    #grid.grid_eq = weights * grid.rho * (1 + grid.cx * grid.u_vec[:, :, 0])
    pass


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