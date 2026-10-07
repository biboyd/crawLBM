"""
Example plot: snapshots of the impulsively-started Couette profile as it
relaxes toward the steady linear profile.

For a single resolution, the simulation is run once and the u_x(y) profile
is captured (via linear interpolation between bracketing lattice steps, as
in convergence_study.py) at several dimensionless times tau_star =
nu*t/H^2. Each snapshot is plotted together with the matching point on the
analytic Fourier-series transient solution, so agreement at each stage of
the approach to steady state is visible directly, with the analytic steady
(linear) profile shown as a reference.

Usage: python plot_snapshots.py
"""
import numpy as np
import matplotlib.pyplot as plt

from crawlbm.collide import calc_rho, calc_u, do_collision
from crawlbm.stream import do_stream
from convergence_study import (
    _make_couette_grid, analytic_steady_profile, analytic_transient_profile,
)


def run_couette_snapshots(ny, tau_star_list, tau=1.0, U_wall=0.04, nx=4):
    """Run once, capturing u_x(y) at each tau_star in tau_star_list via
    linear interpolation between the bracketing lattice steps."""
    nu_lat = (tau - 0.5) / 3.0
    grid = _make_couette_grid(ny, nx, U_wall)

    targets = sorted(tau_star_list)
    target_steps = [t * ny ** 2 / nu_lat for t in targets]
    max_steps = int(np.ceil(target_steps[-1])) + 1

    profiles = {}
    idx = 0
    prev_profile = grid.uvec[:, 0, 0].copy()
    if targets[0] <= 0.0:
        profiles[targets[0]] = prev_profile.copy()
        idx = 1

    for step in range(1, max_steps + 1):
        do_collision(grid, tau)
        do_stream(grid)
        calc_rho(grid)
        calc_u(grid)
        curr_profile = grid.uvec[:, 0, 0].copy()

        while idx < len(targets) and step >= target_steps[idx]:
            frac = target_steps[idx] - (step - 1)
            profiles[targets[idx]] = (1 - frac) * prev_profile + frac * curr_profile
            idx += 1

        prev_profile = curr_profile

    return profiles


def main():
    ny = 64
    tau = 1.0
    U_wall = 0.04
    tau_star_list = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]

    profiles = run_couette_snapshots(ny, tau_star_list, tau=tau, U_wall=U_wall)

    y_over_H = (np.arange(ny) + 0.5) / ny
    cmap = plt.get_cmap('Blues')
    colors = cmap(np.linspace(0.4, 1.0, len(tau_star_list)))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for color, t_star in zip(colors, sorted(tau_star_list)):
        sim = profiles[t_star]
        ax.plot(sim / U_wall, y_over_H, color=color, lw=1.2,
                 label=fr'$t^*={t_star:g}$')
        analytic = analytic_transient_profile(ny, t_star)
        ax.plot(analytic, y_over_H, '--', color=color, lw=2, alpha=0.8)

    steady = analytic_steady_profile(ny, U_wall) / U_wall
    ax.plot(steady, y_over_H, ':', color='black', lw=1.8, label='steady (analytic)')

    ax.set_xlabel(r'$u_x\,/\,U_{wall}$')
    ax.set_ylabel(r'$y\,/\,H$')
    ax.set_title('Impulsively-started Couette flow: approach to steady state\n'
                 '(solid = simulation, dashed = analytic)')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('couette_profile_snapshots.png', dpi=150)
    print('Saved couette_profile_snapshots.png')


if __name__ == '__main__':
    main()
