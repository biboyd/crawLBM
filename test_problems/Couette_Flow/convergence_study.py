"""
Convergence study for plane Couette flow.

Part 1: the *steady* Couette profile is linear, so it has zero
curvature and the D2Q9 BGK scheme reproduces it essentially exactly at any
resolution (error pinned at the steady-state iteration tolerance, not a
discretization error) -- there is no truncation error left to converge away.
This is reported as a correctness check, not a convergence order.

Part 2 (the actual convergence study): the *impulsively-started* Couette flow
(bottom wall at rest, top wall suddenly set to U_wall at t=0) has a curved,
time-dependent profile with a known analytic (Fourier sine series) solution,
so it does carry genuine spatial/temporal truncation error. Comparing the
simulated profile to that series solution at a fixed dimensionless time
nu*t/H^2, across increasing resolution (with the diffusive time step count
scaling as ny^2 so the dimensionless time is matched), shows the error
shrinking with resolution at the scheme's expected order.

Note on the analytic comparison: simple (on-site) bounceback places the
no-slip wall a half lattice unit beyond the last fluid node, so the
effective channel height is H = ny lattice units and fluid row j sits at
y/H = (j + 0.5) / ny.

Usage: python convergence_study.py
"""
import numpy as np
import matplotlib.pyplot as plt

from crawlbm.grid import Grid
from crawlbm.collide import calc_feq, calc_rho, calc_u, do_collision
from crawlbm.stream import do_stream


def _make_couette_grid(ny, nx, U_wall):

    # init grid at equilib
    grid = Grid(nx, ny)
    grid.rho[:] = 1.0
    grid.uvec[:] = 0.0
    calc_feq(grid)
    grid.grid = np.copy(grid.grid_eq)

    # set bound conditions
    grid.bc_vertical = ['periodic', 'periodic']
    grid.bc_horizontal = ['bounceback', 'bounceback']
    grid.bc_horizontal_kwarg = dict(Uwall=[[0., 0.], [U_wall, 0.]])

    return grid


def run_couette_steady(ny, tau=1.0, U_wall=0.01, nx=4, n_diffusion_times=4.0, tol=1e-10):
    """Run plane Couette flow to steady state."""
    nu_lat = (tau - 0.5) / 3.0
    grid = _make_couette_grid(ny, nx, U_wall)

    # set runtime
    max_steps = int(n_diffusion_times * ny**2 / nu_lat)
    check_int = max(20, max_steps // 500)

    # run sim
    prev_profile = None
    step = 0
    for step in range(max_steps):
        do_collision(grid, tau)
        do_stream(grid)

        if step % check_int == 0:
            calc_rho(grid)
            calc_u(grid)

            # check if converged
            profile = grid.uvec[:, 0, 0].copy()
            if prev_profile is not None and np.max(np.abs(profile - prev_profile)) < tol:
                break
            prev_profile = profile

    # return ux(x=0, y) profile
    calc_rho(grid)
    calc_u(grid)
    return grid.uvec[:, 0, 0], step


def analytic_steady_profile(ny, U_wall):
    j = np.arange(ny)
    return U_wall * (j + 0.5) / ny


def run_couette_transient(ny, tau_star, tau=1.0, U_wall=0.01, nx=4):
    """
    Run impulsively-started plane Couette flow up to dimensionless diffusion
    time tau_star = nu_lat * t / H**2 (H = ny), returning the velocity
    profile at that time.

    Since t is only defined at integer lattice steps, the target time
    generally falls between two steps; the profile is linearly interpolated
    between the bracketing steps so that timing round-off (which would
    otherwise dominate the error of this transient, sharply-varying problem)
    isn't mistaken for spatial discretization error.
    """
    nu_lat = (tau - 0.5) / 3.0
    grid = _make_couette_grid(ny, nx, U_wall)

    # set runtime
    t_target = tau_star * ny**2 / nu_lat
    n_lo = int(np.floor(t_target))
    frac = t_target - n_lo

    # run sim
    for _ in range(n_lo):
        do_collision(grid, tau)
        do_stream(grid)

    # calc final fraction tstep
    calc_rho(grid)
    calc_u(grid)
    profile_lo = grid.uvec[:, 0, 0].copy()

    do_collision(grid, tau)
    do_stream(grid)
    calc_rho(grid)
    calc_u(grid)
    profile_hi = grid.uvec[:, 0, 0].copy()

    profile = (1 - frac) * profile_lo + frac * profile_hi
    return profile, t_target


def analytic_transient_profile(ny, tau_star, n_terms=500):
    """
    Fourier series solution for impulsively-started plane Couette flow
    (bottom wall at rest, top wall suddenly started at U at t=0):

        u(y,t)/U = y/H + (2/pi) * sum_n [1 / n] sin(n*pi*(1-y/H))
                          * exp(-n^2 * pi^2 * nu*t/H^2)

    with y/H evaluated at fluid-node locations (j + 0.5)/ny.
    """
    y_over_H = (np.arange(ny) + 0.5) / ny
    u = y_over_H.copy()
    for n in range(1, n_terms + 1):
        u = u - (2.0 / (np.pi * n)) \
            * np.sin(n * np.pi * (1-y_over_H)) * np.exp(-(n ** 2) * (np.pi ** 2) * tau_star)
    return u


def main():
    tau = 1.0
    U_wall = 0.04
    resolutions = np.array([8, 16, 32, 64, 128])

    # --- Part 1: steady-state sanity check -----------------------------
    print("Steady-state check (linear profile; expect near-exact match at all ny):")
    for ny in resolutions:
        sim, nsteps = run_couette_steady(int(ny), tau=tau, U_wall=U_wall)
        analytic = analytic_steady_profile(ny, U_wall)
        err = np.sqrt(np.mean((sim - analytic) ** 2)) / U_wall
        print(f"  ny={ny:4d}  steps={nsteps:7d}  rms_err/U_wall={err:.4e}")

    # --- Part 2: transient convergence study ----------------------------
    tau_star = 0.1  # mid-development: profile is visibly curved, not yet steady
    print(f"\nTransient convergence study at fixed dimensionless time "
          f"nu*t/H^2 = {tau_star}:")

    # decrease Uwall to maintain const Re, nu
    wall_arr = U_wall * 0.5**(np.arange(0, len(resolutions)))
    errors = []
    for ny,curr_U_wall in zip(resolutions, wall_arr):
        
        sim, t_target = run_couette_transient(int(ny), tau_star, tau=tau, U_wall=curr_U_wall)
        analytic = analytic_transient_profile(int(ny), tau_star) * curr_U_wall
        err = np.sqrt(np.mean((sim - analytic) ** 2)) / curr_U_wall
        errors.append(err)
        print(f"  ny={ny:4d}  t_target={t_target:9.2f}  rms_err/U_wall={err:.4e}")

    errors = np.array(errors)

    # calc order of convergence
    orders = np.log(errors[:-1] / errors[1:]) / np.log(resolutions[1:] / resolutions[:-1])
    print("\nObserved local convergence order:")
    for n0, n1, p in zip(resolutions[:-1], resolutions[1:], orders):
        print(f"  {n0:4d} -> {n1:4d}: order = {p:.2f}")

    # plot convergence
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(resolutions, errors, 'o-', label='measured error')
    ref2 = errors[0] * (resolutions[0] / resolutions) ** 2
    ax.loglog(resolutions, ref2, 'k--', label='2nd-order reference')
    ax.set_xlabel('ny (grid points across channel)')
    ax.set_ylabel(r'RMS($u_x$ error) / $U_{wall}$')
    ax.set_title(f'Impulsively-started Couette flow: convergence at '
                 fr'$\nu t/H^2={tau_star}$')
    ax.legend()
    fig.tight_layout()
    fig.savefig('couette_convergence.png', dpi=150)
    print("\nSaved couette_convergence.png")


if __name__ == '__main__':
    main()
