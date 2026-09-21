"""
Will use this to init a grid and 
fill distribution functions
"""

import ArgumentParser from argparse
from grid import Grid
from collide import calc_feq
from advance import advance_sim

def run_sim(parser):

    nx, ny = parser.domain_size
    init_grid = Grid(nx, ny)

    # init vel field
    if parser.init == 'TG-vortex':
        init_TG_vortex(init_grid)
    else:
        raise NotImplementedError(f"{parser.init} has not has been implemented. use default")

    advance_sim(init_grid, parser.tau, max_step=parser.max_step)

def init_TG_vortex(grid, k=2*np.pi, rho0=1., U_0=1.):

    x_arr = np.linspace(0, 1, grid.nx)
    y_arr = np.linspace(0, 1, grid.ny)

    # set init velocity
    grid.uvec[:, :, 0] = -U_0 * np.cos(k * x_arr) * np.sin(k * y_arr)
    grid.uvec[:, :, 1] = U_0 * np.sin(k * x_arr) * np.cos(k * y_arr)

    # set init pressure
    p_0 = rho_0 * U_0**2 * (np.cos(2*k*x_arr) + np.cos(2*k*y_arr)) / 4.
    p_avg = np.mean(p_0) 

    init_f_rho(grid, p_0)

def init_f_rho(grid, p_0):
    """
    sets the init distributions f's and the density. Assumes a set p0 and that 
    the velocities grid.uvec have been initialized
    """
    # set avg density (this is essentially just applying EOS)
    cs2_inv = 3.
    grid.rho = rho0 + cs2_inv*(p_0 - p_avg)

    # set f's at equilib
    calc_feq(grid)
    grid.grid_eq = grid.grid


if __init__ == '__main__':

    parser = ArgumentParser(
                        prog='CrawLBM',
                        description='Runs CrawLBM which will (very slowly) \
                                     evolve a fluid field using a LB method',
                        )

    #parser.add_argument('--inputs', help='inputs file for the arg')

    parser.add_argument('--domain_size', nargs=2, type=int, default=[10, 10], 
                        help='Size of domain size in x and y dir')

    parser.add_argument('--init', type=str, default='TG-vortex',
                        help='What sort of initialization to use. \
                              types include "TG-vortex", more to be added.')

    parser.add_argument('--tau', type=float, default=0.5, help='Relaxation time.')

    parser.add_argument('--max_step', type=float, default=1e3, help='Max number of timesteps before stopping')

    run_sim(parser)
