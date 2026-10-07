"""
Example plots: evolution of the 2D Taylor-Green vortex decay.

1. Heatmap snapshots of u_x at several dimensionless times t_star =
   nu*(kx^2+ky^2)*t, showing the vortex pattern decaying in place (the
   spatial pattern doesn't change -- only its amplitude does, since the
   single-mode TG solution decays self-similarly).
2. The spatially-averaged (RMS) speed vs. time, compared against the
   analytic exponential decay -- since every point decays by the same
   factor exp(-t_star), any spatial norm of the velocity field decays as a
   pure exponential.

Usage: python plot_snapshots.py
"""
import numpy as np
import matplotlib.pyplot as plt

from crawlbm.collide import calc_rho, calc_u, do_collision
from crawlbm.stream import do_stream
from convergence_study import init_tg_grid


def run_tg_trace(n, tau, U_0, t_star_max, heatmap_times, n_samples=300):
    """Run the TG vortex decay once, recording the RMS speed at ~n_samples
    points up to t_star_max, and the full u_x field at each time in
    heatmap_times."""
    nu_lat = (tau - 0.5) / 3.0
    grid, kx, ky = init_tg_grid(n, n, U_0=U_0)
    t_c_inv = nu_lat * (kx ** 2 + ky ** 2)

    steps_max = int(np.ceil(t_star_max / t_c_inv))
    sample_every = max(1, steps_max // n_samples)
    heatmap_steps = {int(round(t / t_c_inv)) for t in heatmap_times}

    calc_rho(grid)
    calc_u(grid)
    t_trace = [0.0]
    speed_trace = [np.sqrt(np.mean(grid.uvec[:, :, 0] ** 2 + grid.uvec[:, :, 1] ** 2))]
    heatmaps = {}
    if 0 in heatmap_steps:
        heatmaps[0.0] = grid.uvec[:, :, 0].copy()

    for step in range(1, steps_max + 1):
        do_collision(grid, tau)
        do_stream(grid)

        if step % sample_every == 0 or step in heatmap_steps or step == steps_max:
            calc_rho(grid)
            calc_u(grid)
            t_star = step * t_c_inv
            t_trace.append(t_star)
            speed_trace.append(
                np.sqrt(np.mean(grid.uvec[:, :, 0] ** 2 + grid.uvec[:, :, 1] ** 2))
            )
            if step in heatmap_steps:
                heatmaps[round(t_star, 4)] = grid.uvec[:, :, 0].copy()

    return np.array(t_trace), np.array(speed_trace), heatmaps


def plot_heatmaps(heatmaps, U_0):
    times_sorted = sorted(heatmaps.keys())

    fig, axes = plt.subplots(1, len(times_sorted),
                              figsize=(3.0 * len(times_sorted), 3.3),
                              constrained_layout=True)
    im = None
    for ax, t_star in zip(axes, times_sorted):
        im = ax.imshow(heatmaps[t_star] / U_0, cmap='RdBu', vmin=-1, vmax=1,
                        origin='lower')
        ax.set_title(fr'$t^*={t_star:.2f}$', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, label=r'$u_x\,/\,U_0$')
    fig.suptitle('2D Taylor-Green vortex: decay of $u_x$')
    fig.savefig('tgv_heatmap_snapshots.png', dpi=150)
    print('Saved tgv_heatmap_snapshots.png')


def plot_mean_speed(t_trace, speed_trace, U_0):
    analytic = speed_trace[0] * np.exp(-t_trace)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(t_trace, speed_trace / U_0, color='tab:blue', lw=2, label='simulation')
    ax.plot(t_trace, analytic / U_0, '--', color='black', lw=2, label='analytic decay')

    ax.set_xlabel(r'dimensionless time $t^* = \nu (k_x^2+k_y^2)\,t$')
    ax.set_ylabel(r'RMS speed $\sqrt{\langle u_x^2+u_y^2\rangle}\,/\,U_0$')
    ax.set_title('Taylor-Green vortex: spatially-averaged speed vs. time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('tgv_mean_speed_vs_time.png', dpi=150)
    print('Saved tgv_mean_speed_vs_time.png')


def main():
    n = 128
    tau = 0.9
    U_0 = 0.03
    t_star_max = 3.0
    heatmap_times = [0.0, 0.3, 0.8, 1.5, 3.0]

    t_trace, speed_trace, heatmaps = run_tg_trace(n, tau, U_0, t_star_max, heatmap_times)

    plot_heatmaps(heatmaps, U_0)
    plot_mean_speed(t_trace, speed_trace, U_0)


if __name__ == '__main__':
    main()
