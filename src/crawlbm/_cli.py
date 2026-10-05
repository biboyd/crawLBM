"""
Will use this to init a grid and
fill distribution functions
"""

import os
from argparse import ArgumentParser
from os import makedirs
import numpy as np

from .grid import Grid
from .collide import calc_feq, do_init_collision, calc_u, calc_rho
from .stream import do_stream
from .advance import advance_sim

def run_sim(args):


    # set grid size and spacing
    Lx, Ly = args.domain_size
    grid_spacing = max(Lx, Ly)/args.domain_resolution
    nx = int(Lx/grid_spacing)
    ny = int(Ly/grid_spacing)

    # calc Re number
    Re = args.U_ref * max(Lx, Ly)/args.nu

    # calc lattice viscosity
    nu_lat = (1./3.) * (args.tau - 0.5) # assuming cs^2= 1/3

    # calc conversion factors
    C_x = grid_spacing # this sets del_x_phy assuming del_x_lat = 1.
    C_t = nu_lat * C_x**2/args.nu # this set del_t_phys assuming del_t_lat = 1.
    #C_rho = args.rho_ref / 1.0 # assuming rho_lat = 1

    U_lat = args.U_ref * C_t/C_x
    Ma = U_lat / np.sqrt(3)

    # print out params
    VERBOSE = True
    # Verify constraints before any derived computation that assumes tau > 0.5
    if args.tau <= 0.5:
        raise ValueError(f"tau {args.tau} <= 0.5! Increase tau for positive viscosity.")
    if Ma >= 0.1:
        raise ValueError(f"Mach number {Ma:.4f} >= 0.1! Reduce U_ref or increase resolution.")
    if args.tau >= 1.8:
        print(f"Warning: tau {args.tau} >= 1.8 may cause numerical instability.")

    if VERBOSE:
        print(f"nu_phy: {args.nu} m^2/s")
        print(f"Timestep: {C_t} s")
        print(f"Spacing: {C_x} m")
        print(f"nu_lat: {nu_lat} lu^2/ts")
        print(f"U_lat: {U_lat} lu/ts")
        print(f"Re_lat: {U_lat/nu_lat} ")
        print(f"Re: {Re}")
        print(f"Ma: {Ma}")

    # setup plot dir early so analytic solutions land in the same place
    makedirs(args.plot_dir, exist_ok=True)

    init_grid = Grid(nx, ny)

    # init vel field
    if args.init == 'TG-vortex':
        init_TG_vortex(init_grid, Lx, Ly, U_0=args.U_ref)

        # run analytic soln
        analytic_TG_vortex(nx, ny, Lx, Ly, args.nu, args.max_step, args.plot_int,
                            dt_phy=C_t, U_0=args.U_ref, plot_dir=args.plot_dir)

    elif args.init == 'Couette':
        init_Couette_flow(init_grid, U_0=args.U_ref)

        # run analytic soln
        analytic_Couette_flow(nx, ny, Lx, Ly, U_0=args.U_ref, plot_dir=args.plot_dir)
    else:
        raise NotImplementedError(f"{args.init} has not has been implemented. use default")

    # relax distribution prior to running sim
    run_initialization(init_grid, args.init_steps, args.tau)

    # run sim
    advance_sim(init_grid, args.tau, dt_phy = C_t, max_step=args.max_step,
                plot_int=args.plot_int, plot_dir=args.plot_dir)


def init_TG_vortex(grid, Lx, Ly, rho_0=1., U_0=0.01, VERBOSE=False):

    # calc const
    kx = 2*np.pi/Lx
    ky = 2*np.pi/Ly

    # create x-y mesh
    x_axis = np.linspace(0, Lx, grid.nx)
    y_axis = np.linspace(0, Ly, grid.ny)
    x_arr, y_arr = np.meshgrid(x_axis, y_axis)

    # set BC
    grid.bc_vertical = ['periodic', 'periodic']
    grid.bc_horizontal = ['periodic', 'periodic']

    # set init velocity
    grid.uvec[:, :, 0] = -U_0 * np.sqrt(ky/kx) * np.cos(kx * x_arr) * np.sin(ky * y_arr)
    grid.uvec[:, :, 1] = U_0 * np.sqrt(kx/ky) * np.sin(kx * x_arr) * np.cos(ky * y_arr)

    if VERBOSE:
        import matplotlib.pyplot as plt
        plt.imshow(grid.uvec[:, :, 0])
        plt.savefig("init_ux.png")

    # set init pressure
    p_0 = -rho_0 * U_0**2 * (np.cos(2*kx*x_arr) + np.cos(2*ky*y_arr)) / 4.
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

def init_Couette_flow(grid, rho_0=1., U_0=0.01):
    # set rho and U
    grid.rho = np.ones((grid.ny, grid.nx))*rho_0
    grid.uvec = np.zeros((grid.ny, grid.nx, 2))

    # set grid to eq
    calc_feq(grid)
    grid.grid = np.copy(grid.grid_eq)

    # set BCs
    grid.bc_vertical = ['periodic', 'periodic']
    grid.bc_horizontal = ['bounceback', 'bounceback']

    grid.bc_horizontal_kwarg = dict(Uwall=[[0., 0.],
                                           [U_0, 0.]])

def analytic_TG_vortex(nx, ny, Lx, Ly, nu, max_step, plot_int, dt_phy=1., U_0=0.01, plot_dir='./'):
    # calc viscous time
    kx = 2 * np.pi / Lx
    ky = 2 * np.pi / Ly
    t_c_inv = nu * (kx**2 +ky**2)

    # create x-y mesh
    x_axis = np.linspace(0, Lx, nx)
    y_axis = np.linspace(0, Ly, ny)
    x_arr, y_arr = np.meshgrid(x_axis, y_axis)

    uvec = np.ndarray((ny, nx, 2))
    # set init velocity
    uvec[:, :, 0] = -U_0 * np.cos(kx * x_arr) * np.sin(ky * y_arr)
    uvec[:, :, 1] = U_0 * np.sin(kx * x_arr) * np.cos(ky * y_arr)

    for i in range(int(max_step)):

        if i % plot_int == 0:
            time = dt_phy * (i+1)
            #calc u's for curr time
            curr_ux = uvec[:, :, 0] * np.exp(-time*t_c_inv)
            curr_uy = uvec[:, :, 1] * np.exp(-time*t_c_inv)

            #save plotfile
            outfile = os.path.join(plot_dir, f"analytic{i:07d}.npy")
            out_arr = np.dstack((curr_ux, curr_uy))
            np.save(outfile, out_arr)

def analytic_Couette_flow(nx, ny, Lx, Ly, U_0=0.01, plot_dir='./'):
    # the analytic soln should just be linear flow
    # create x-y mesh
    x_axis = np.linspace(0, Lx, nx)
    y_axis = np.linspace(0, Ly, ny)
    x_arr, y_arr = np.meshgrid(x_axis, y_axis)

    # construct velocity arr
    u_vec = np.empty((ny, nx, 2))

    # Ux linear in y
    u_vec[:, :, 0] = y_arr * U_0/Ly

    # Uy zero
    u_vec[:, :, 1] = 0.

    #save plotfile
    outfile = os.path.join(plot_dir, "analytic_solution.npy")
    np.save(outfile, u_vec)

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
    WIP

    once we init a distribution and velocity we need to then relax
    to an appropriate solution. Do this by just repeating stream/collide basically
    but keeping vel0 constant and wanting to end on a propogate so start of sim is
    a collision.

    think this is accurate but I may be doing things incorrectly. will find out

    essentially do the collide propagate but keep velocity fixed at all times.
    """

    pass
    # WIP
    #for i in range(int(Nsteps)):
    #    # collide
    #    do_init_collision(grid, tau, dt)

    #    # propogate/stream
    #    do_stream(grid)


def main():
    parser = ArgumentParser(
                        prog='crawlbm',
                        description='Runs CrawLBM which will (very slowly) \
                                     evolve a fluid field using a LB method',
                        )

    parser.add_argument('--domain_size', nargs=2, type=float, default=[1, 1],
                        help='Size of domain size in x and y dir (SI Units)')

    parser.add_argument('--domain_resolution', type=int, default=64,
                        help='resolution in longest direction')

    parser.add_argument('--init', type=str, default='TG-vortex',
                        help='What sort of initialization to use. \
                              types include "TG-vortex", more to be added.')

    parser.add_argument('--init_steps', type=float, default=100, help='number of times to relax init field')

    parser.add_argument('--tau', type=float, default=0.8, help='Relaxation time. must be greater than 0.5')
    parser.add_argument('--nu', type=float, default=1e-6, help='Physical viscosity (SI units)')
    parser.add_argument('--U_ref', type=float, default=0.01, help='Physical velocity scale. Sets U_0 depending on init setups (SI units).')
    parser.add_argument('--rho_ref', type=float, default=1., help='Physical density scale. Sets rho_0 depending on init setups (SI units).')

    parser.add_argument('--max_step', type=float, default=1e3, help='Max number of timesteps before stopping.')
    parser.add_argument('--plot_int', type=int, default=100, help='Save a plotfile for every given timesteps.')
    parser.add_argument('--plot_dir', type=str, default='./', help='directory to save plotfiles in.')

    args = parser.parse_args()
    run_sim(args)


if __name__ == '__main__':
    main()
