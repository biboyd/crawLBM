"""
Unit tests for core LBM functions: streaming, collision, and boundary conditions.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from grid import Grid
from stream import do_stream
from collide import calc_feq, do_collision


def _make_grid(nx=6, ny=6):
    """Return a zeroed Grid with rho/uvec pre-initialized."""
    g = Grid(nx, ny)
    g.rho = np.ones((ny, nx))
    g.uvec = np.zeros((ny, nx, 2))
    g.grid[:] = 0.0
    g.grid_eq = np.zeros((ny, nx, g.nf))
    return g


def test_do_stream():
    """
    A distribution placed at [i, j, idx] should appear at
    [i + cx[idx], j + cy[idx]] after one stream step under periodic BCs.

    Direction 5 (cx=1, cy=1) is used because periodic_bc only asserts on
    the four cardinal directions (1-4), keeping the test independent of that check.
    """
    g = _make_grid()
    g.bc_vertical = ['periodic', 'periodic']
    g.bc_horizontal = ['periodic', 'periodic']

    i, j = 2, 3
    g.grid[i, j, 5] = 1.0  # f_5: cx=1, cy=1

    do_stream(g)

    assert g.grid[i + 1, j + 1, 5] == 1.0, "f_5 did not reach (i+1, j+1)"
    assert g.grid[i, j, 5] == 0.0, "f_5 should have vacated the source cell"


def test_calc_feq():
    """
    At unit density and zero velocity the equilibrium distribution must equal
    the lattice weights: f_eq[i] = w[i] for every direction i.
    """
    g = _make_grid()
    g.rho = np.ones((g.ny, g.nx))
    g.uvec = np.zeros((g.ny, g.nx, 2))

    calc_feq(g)

    for idx in range(g.nf):
        np.testing.assert_allclose(
            g.grid_eq[:, :, idx], g.weights[idx], rtol=1e-12,
            err_msg=f"f_eq direction {idx}: expected {g.weights[idx]}"
        )


def test_do_collision():
    """
    A grid already at equilibrium (f = f_eq) must be invariant under collision
    for any relaxation time tau, because (1-omega)*f_eq + omega*f_eq = f_eq.
    """
    g = _make_grid()
    for idx in range(g.nf):
        g.grid[:, :, idx] = g.weights[idx]

    tau = 0.7
    do_collision(g, tau)

    for idx in range(g.nf):
        np.testing.assert_allclose(
            g.grid[:, :, idx], g.weights[idx], rtol=1e-12,
            err_msg=f"Collision changed direction {idx} away from equilibrium"
        )


def test_periodic_bc():
    """
    Under periodic BCs, a distribution that exits one boundary must re-enter
    from the opposite boundary.  f_2 (cx=0, cy=1) is chosen because
    periodic_bc explicitly asserts its wrap-around along axis=1.
    """
    g = _make_grid(nx=6, ny=6)
    g.bc_vertical = ['periodic', 'periodic']
    g.bc_horizontal = ['periodic', 'periodic']

    # f_2 at the last column; after rolling axis=1 by cy[2]=1 it should wrap to column 0
    g.grid[2, -1, 2] = 1.0

    do_stream(g)

    assert g.grid[2, 0, 2] == 1.0, "f_2 did not wrap from last column to column 0"
    assert g.grid[2, -1, 2] == 0.0, "f_2 should have left the last column"


def test_bounceback_bc():
    """
    With a stationary bounceback wall at axis=0 row 0, downward-moving
    distributions (f_4, d_i=4) are reflected as upward-moving distributions
    (f_2, u_i=2) at the same boundary row, with zero wall contribution.
    """
    g = _make_grid(nx=6, ny=6)
    g.bc_vertical = ['periodic', 'periodic']
    g.bc_horizontal = ['bounceback', 'periodic']  # bounceback only at row 0

    val = 0.5
    g.grid[0, :, 4] = val  # f_4 (cx=0, cy=-1) sitting at the wall row

    do_stream(g)

    # horizontal_wall pair (d_i=4, u_i=2); wall_term=0 for stationary wall
    np.testing.assert_allclose(
        g.grid[0, :, 2], val, rtol=1e-12,
        err_msg="f_4 was not reflected back as f_2 by the bounceback wall"
    )


if __name__ == '__main__':
    tests = [test_do_stream, test_calc_feq, test_do_collision,
             test_periodic_bc, test_bounceback_bc]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
