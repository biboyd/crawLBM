"""
compute the collision step w/ a BGK operator
"""

import numpy as np

def calc_rho(grid):
    """
    calc rho at every point on the grid
    """

    grid.rho = np.sum(grid.grid, axis=2)

def calc_u(grid):
    """
    calc u at every point on the grid
    """

    # construct vector for u

    # should prolly unroll this. doing lots of unncessary multiplying by zero
    grid.uvec[:, :, 0] = np.matvec(grid.grid, grid.cx) / grid.rho
    grid.uvec[:, :, 1] = np.matvec(grid.grid, grid.cy) / grid.rho

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

    vel_term = 1. + cs2_inv * (np.tensordot(grid.uvec[:, :, 0], grid.cx, 0) +  \
                               np.tensordot(grid.uvec[:, :, 1], grid.cy, 0) ) \
               + cs4_2_inv * (np.tensordot(grid.uvec[:, :, 0]**2,grid.cx**2 - 1./cs2_inv, 0) + \
                              np.tensordot(grid.uvec[:, :, 1]**2,grid.cy**2 - 1./cs2_inv, 0) + \
                              np.tensordot(2.*grid.uvec[:, :, 0]*grid.uvec[:, :, 1], grid.cx*grid.cy, 0))
    grid.grid_eq = np.tensordot(grid.rho, grid.weights, 0) * vel_term


def do_collision(grid, tau, dt):
    """
    calculates the collision on the grid and 
    updates the grid.

    Takes in grid (type Grid) and tau (relaxation time) and timestep, dt, I guess
    """

    calc_rho(grid)
    calc_u(grid)

    calc_feq(grid)
    omega = dt/tau

    grid.grid = (1-omega) * grid.grid + omega * grid.grid_eq

def do_init_collision(grid, tau, dt):
    """
    calculates the collision on the grid and 
    updates the grid.

    Takes in grid (type Grid) and tau (relaxation time) and timestep, dt, I guess
    """

    calc_rho(grid)
    # don't update uvec so always the init

    calc_feq(grid)
    omega = dt/tau

    grid.grid = (1-omega) * grid.grid + omega * grid.grid_eq

