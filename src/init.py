"""
Will use this to init a grid and 
fill distribution functions
"""

from argparse import ArgumentParser 
from grid import Grid
from collide import calc_feq, do_init_collision,calc_u, calc_rho
from stream import do_stream
from advance import advance_sim
import numpy as np

def run_sim(args):

    nx, ny = args.domain_size
    init_grid = Grid(nx, ny)

    # init vel field
    if args.init == 'TG-vortex':
        init_TG_vortex(init_grid)

        # run analytic soln
        analytic_TG_vortex(nx, ny, args.tau, max_step=args.max_step, plot_int=args.plot_int)

    elif args.init == 'Couette':
        init_Couette_flow(init_grid)

        # run analytic soln
        analytic_Couette_flow(nx, ny, args.tau, max_step=args.max_step, plot_int=args.plot_int)
    else:
        raise NotImplementedError(f"{args.init} has not has been implemented. use default")

    # relax distribution prior to running sim
    run_initialization(init_grid, args.init_steps, args.tau)

    # run sim
    advance_sim(init_grid, args.tau, max_step=args.max_step, plot_int=args.plot_int)


def init_TG_vortex(grid, k=2*np.pi, rho_0=1., U_0=0.01, VERBOSE=False):


    # create x-y mesh
    x_axis = np.linspace(0, 1, grid.nx)
    y_axis = np.linspace(0, 1, grid.ny)
    x_arr, y_arr = np.meshgrid(x_axis, y_axis)

    # set BC
    grid.bc_vertical = ['periodic', 'periodic']
    grid.bc_horizontal = ['periodic', 'periodic']

    # set init velocity
    grid.uvec[:, :, 0] = -U_0 * np.cos(k * x_arr) * np.sin(k * y_arr)
    grid.uvec[:, :, 1] = U_0 * np.sin(k * x_arr) * np.cos(k * y_arr)

    if VERBOSE:
        import matplotlib.pyplot as plt
        plt.imshow(grid.uvec[:, :, 0])
        plt.savefig("init_ux.png")

    # set init pressure
    p_0 = -rho_0 * U_0**2 * (np.cos(2*k*x_arr) + np.cos(2*k*y_arr)) / 4.
    p_avg = np.mean(p_0) 

    if VERBOSE:
        import matplotlib.pyplot as plt
        plt.imshow(p_0)
        plt.savefig("init_p0.png")


    init_f_rho(grid, p_0, rho_0, p_avg)

    # check
    calc_rho(grid)
    calc_u(grid)
    if VERBOSE:
        import matplotlib.pyplot as plt
        plt.imshow(grid.uvec[:, :, 0])
        plt.savefig("after_calc_ux.png")

        plt.imshow(grid.rho)
        plt.savefig("after_init_rho.png")

def init_Couette_flow(grid, rho_0=1.):
    # set rho and U
    grid.rho = np.ones((grid.nx, grid.ny))*rho_0
    grid.uvec = np.zeros((grid.nx, grid.ny, 2))

    # set grid to eq
    calc_feq(grid)
    grid.grid = np.copy(grid.grid_eq)

    # set BCs
    grid.bc_vertical = ['periodic', 'periodic']
    grid.bc_horizontal = ['bounceback', 'bounceback']

    grid.bc_kwarg = dict(Uwall=[0., 0.01])

def analytic_TG_vortex(nx, ny, tau, max_step, plot_int):
    # hard coded variables
    k=2*np.pi; rho_0=1.; U_0=0.01
    delx = 1/nx
    nu = 1/3. * (tau - 1./2)*delx**2 # assuming dt=1 and cs^2= 1/3
    print(f"Viscosity nu= {nu:0.2e}")
    print(f"Re= {U_0 / nu:0.2e}")
    t_c_inv = 2*nu * k**2 

    # create x-y mesh
    x_axis = np.linspace(0, 1, nx)
    y_axis = np.linspace(0, 1, ny)
    x_arr, y_arr = np.meshgrid(x_axis, y_axis)

    uvec = np.ndarray((nx, ny, 2))
    # set init velocity
    uvec[:, :, 0] = -U_0 * np.cos(k * x_arr) * np.sin(k * y_arr)
    uvec[:, :, 1] = U_0 * np.sin(k * x_arr) * np.cos(k * y_arr)

    for i in range(int(max_step)):

        if i % plot_int == 0:
            time = i+1
            #calc u's for curr time
            curr_ux = uvec[:, :, 0] * np.exp(-time*t_c_inv)
            curr_uy = uvec[:, :, 1] * np.exp(-time*t_c_inv)

            #save plotfile
            outfile = f"analytic{i:07d}.npy"
            out_arr = np.dstack((curr_ux, curr_uy))
            np.save(outfile, out_arr)
            
def analytic_Couette_flow(nx, ny, tau, max_step, plot_int, U_0=0.01):
    # the analytic soln should just be linear flow
    # create x-y mesh
    x_axis = np.linspace(0, 1, nx)
    y_axis = np.linspace(0, 1, ny)
    x_arr, y_arr = np.meshgrid(x_axis, y_axis)

    # set velocity
    # Ux linear in y
    u_vec = np.empty((nx, ny, 2))
    u_vec[:, :, 0] = y_arr * U_0

    # Uy zero
    u_vec[:, :, 1] = 0.

    #save plotfile
    outfile = f"analytic_solution.npy"
    out_arr = u_vec
    np.save(outfile, out_arr)

def init_f_rho(grid, p_0, rho_0, p_avg):
    """
    sets the init distributions f's and the density. Assumes a set p0 and that 
    the velocities grid.uvec have been initialized
    """
    # set avg density (this is essentially just applying EOS)
    cs2_inv = 3.
    grid.rho = rho_0 + cs2_inv*(p_0 - p_avg)

    # set f's at equilib
    calc_feq(grid)
    grid.grid = np.copy(grid.grid_eq)

def run_initialization(grid, Nsteps, tau, dt=1.):
    """
    once we init a distribution and velocity we need to then relax
    to an appropriate solution. Do this by just repeating stream/collide basically
    but keeping vel0 constant and wanting to end on a propogate so start of sim is
    a collision.

    think this is accurate but I may be doing things incorrectly. will find out

    essentially do the collide propagate but keep velocity fixed at all times.
    """

    for i in range(int(Nsteps)):
        # collide
        do_init_collision(grid, tau, dt)

        # propogate/stream
        do_stream(grid)
    

if __name__ == '__main__':

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

    parser.add_argument('--init_steps', type=float, default=100, help='number of times to relax init field')

    parser.add_argument('--tau', type=float, default=0.5, help='Relaxation time.')

    parser.add_argument('--max_step', type=float, default=1e3, help='Max number of timesteps before stopping')
    parser.add_argument('--plot_int', type=int, default=100, help='n steps to plot out')

    args = parser.parse_args()
    run_sim(args)
