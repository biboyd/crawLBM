"""
Grid-convergence study for the 2D Taylor-Green vortex decay.

The single-mode 2D Taylor-Green vortex

    u_x(x,y,t) = -U0 * cos(kx x) sin(ky y) * exp(-nu (kx^2+ky^2) t)
    u_y(x,y,t) =  U0 * sin(kx x) cos(ky y) * exp(-nu (kx^2+ky^2) t)

is an *exact* solution of the incompressible Navier-Stokes equations (the
nonlinear advection term is balanced by the accompanying pressure field), so
unlike steady plane Couette flow it has nonzero curvature everywhere and a
genuine spatial truncation error for the D2Q9 BGK scheme to converge away, on
top of the scheme's intrinsic O(Ma^2) weakly-compressible error.

The domain is always exactly one wavelength in each direction (kx = ky =
2*pi/n for an n x n grid), and tau (hence nu_lat) is held fixed, so the
viscous decay time nu*(kx^2+ky^2) ~ 1/n^2 shrinks as the grid is refined --
the lattice-step count needed to reach a fixed dimensionless decay time
t_star = nu*(kx^2+ky^2)*t therefore scales as n^2, exactly analogous to the
diffusive-time-scale convergence test used for Couette flow. As in that
script, the simulated field is linearly interpolated between the two
lattice steps bracketing the target time so that time-step round-off isn't
mistaken for spatial discretization error.

Usage: python convergence_study.py
"""
import numpy as np
import matplotlib.pyplot as plt

from crawlbm.grid import Grid
from crawlbm.collide import calc_feq, calc_rho, calc_u, do_collision
from crawlbm.stream import do_stream


def init_tg_grid(n, U_0=0.01, rho_0=1.0):
    """n x n grid, domain = one wavelength per side (kx = ky = 2*pi/n)."""
    grid = Grid(n, n)
    kx = 2 * np.pi / n
    ky = 2 * np.pi / n

    x_arr, y_arr = np.meshgrid(np.arange(n), np.arange(n))

    grid.bc_vertical = ['periodic', 'periodic']
    grid.bc_horizontal = ['periodic', 'periodic']

    grid.uvec[:, :, 0] = -U_0 * np.cos(kx * x_arr) * np.sin(ky * y_arr)
    grid.uvec[:, :, 1] = U_0 * np.sin(kx * x_arr) * np.cos(ky * y_arr)

    p0 = -rho_0 * U_0 ** 2 * (np.cos(2 * kx * x_arr) + np.cos(2 * ky * y_arr)) / 4.
    cs2_inv = 3.
    grid.rho = rho_0 + cs2_inv * (p0 - np.mean(p0))

    calc_feq(grid)
    grid.grid = np.copy(grid.grid_eq)

    return grid, kx, ky


def analytic_tg(n, kx, ky, t_star, U_0=0.01):
    x_arr, y_arr = np.meshgrid(np.arange(n), np.arange(n))
    decay = np.exp(-t_star)
    ux = -U_0 * np.cos(kx * x_arr) * np.sin(ky * y_arr) * decay
    uy = U_0 * np.sin(kx * x_arr) * np.cos(ky * y_arr) * decay
    return ux, uy


def run_tg(n, t_star_target, tau=1.0, U_0=0.01):
    """Run TG vortex decay up to dimensionless time t_star = nu*(kx^2+ky^2)*t."""
    nu_lat = (tau - 0.5) / 3.0
    grid, kx, ky = init_tg_grid(n, U_0=U_0)
    t_c_inv = nu_lat * (kx ** 2 + ky ** 2)

    t_target_steps = t_star_target / t_c_inv
    n_lo = int(np.floor(t_target_steps))
    frac = t_target_steps - n_lo

    for _ in range(n_lo):
        do_collision(grid, tau)
        do_stream(grid)
    calc_rho(grid)
    calc_u(grid)
    ux_lo, uy_lo = grid.uvec[:, :, 0].copy(), grid.uvec[:, :, 1].copy()

    do_collision(grid, tau)
    do_stream(grid)
    calc_rho(grid)
    calc_u(grid)
    ux_hi, uy_hi = grid.uvec[:, :, 0].copy(), grid.uvec[:, :, 1].copy()

    ux = (1 - frac) * ux_lo + frac * ux_hi
    uy = (1 - frac) * uy_lo + frac * uy_hi
    return ux, uy, t_target_steps


def main():
    tau = 1.0
    U_0 = 0.01
    t_star = 1.0
    resolutions = np.array([16, 32, 64, 128, 256])

    print(f"Taylor-Green vortex convergence study at fixed dimensionless time "
          f"nu*(kx^2+ky^2)*t = {t_star}:")

    errors = []
    for n in resolutions:
        n = int(n)
        ux, uy, nsteps = run_tg(n, t_star, tau=tau, U_0=U_0)
        kx = ky = 2 * np.pi / n
        ux_a, uy_a = analytic_tg(n, kx, ky, t_star, U_0=U_0)
        err = np.sqrt(np.mean((ux - ux_a) ** 2 + (uy - uy_a) ** 2)) / U_0
        errors.append(err)
        print(f"  n={n:4d}  steps={nsteps:9.2f}  rms_err/U_0={err:.4e}")

    errors = np.array(errors)

    orders = np.log(errors[:-1] / errors[1:]) / np.log(resolutions[1:] / resolutions[:-1])
    print("\nObserved local convergence order:")
    for n0, n1, p in zip(resolutions[:-1], resolutions[1:], orders):
        print(f"  {n0:4d} -> {n1:4d}: order = {p:.2f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(resolutions, errors, 'o-', label='measured error')
    ref2 = errors[0] * (resolutions[0] / resolutions) ** 2
    ax.loglog(resolutions, ref2, 'k--', label='2nd-order reference')
    ax.set_xlabel('N (grid points per side)')
    ax.set_ylabel(r'RMS velocity error / $U_0$')
    ax.set_title(fr'2D Taylor-Green vortex decay: convergence at '
                 fr'$\nu(k_x^2+k_y^2)t={t_star}$')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig('tgv_convergence.png', dpi=150)
    print("\nSaved tgv_convergence.png")


if __name__ == '__main__':
    main()
