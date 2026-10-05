"""
Unit tests for core LBM functions: streaming, collision, and boundary conditions.

Run with:  pytest unit_tests/test_functions.py -v
       or: python unit_tests/test_functions.py
"""

import numpy as np
import pytest

from crawlbm.grid import Grid
from crawlbm.stream import do_stream
from crawlbm.collide import calc_feq, do_collision


def _make_grid(nx=6, ny=6):
    """Return a zeroed Grid with rho/uvec pre-initialized."""
    g = Grid(nx, ny)
    g.rho = np.ones((ny, nx))
    g.uvec = np.zeros((ny, nx, 2))
    g.grid[:] = 0.0
    g.grid_eq = np.zeros((ny, nx, g.nf))
    return g


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("idx", range(9))
def test_do_stream(idx):
    """
    Every distribution direction shifts by exactly (cx[idx], cy[idx]) in one
    stream step.  The grid is large enough (8x8, tracer at [4,4]) that no
    direction reaches a boundary, so periodic_bc assertions are trivially
    satisfied without wrap-around complicating the check.
    """
    g = _make_grid(nx=8, ny=8)
    g.bc_vertical = ['periodic', 'periodic']
    g.bc_horizontal = ['periodic', 'periodic']

    i0, j0 = 4, 4
    g.grid[i0, j0, idx] = 1.0

    do_stream(g)

    i1 = i0 + g.cx[idx]
    j1 = j0 + g.cy[idx]
    assert g.grid[i1, j1, idx] == 1.0, \
        f"dir {idx} (cx={g.cx[idx]}, cy={g.cy[idx]}): expected 1.0 at [{i1},{j1}]"
    if idx != 0:   # direction 0 is the rest distribution; it stays in place
        assert g.grid[i0, j0, idx] == 0.0, \
            f"dir {idx}: source cell [{i0},{j0}] should be vacated"


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Periodic BC
# ---------------------------------------------------------------------------

def test_periodic_bc():
    """
    Under periodic BCs, a distribution that exits one boundary must re-enter
    from the opposite boundary.  f_2 (cx=0, cy=1) is chosen because
    periodic_bc explicitly asserts its wrap-around along axis=1.
    """
    g = _make_grid(nx=6, ny=6)
    g.bc_vertical = ['periodic', 'periodic']
    g.bc_horizontal = ['periodic', 'periodic']

    # f_2 at the last column; rolling axis=1 by cy[2]=1 wraps it to column 0
    g.grid[2, -1, 2] = 1.0

    do_stream(g)

    assert g.grid[2, 0, 2] == 1.0, "f_2 did not wrap from last column to column 0"
    assert g.grid[2, -1, 2] == 0.0, "f_2 should have left the last column"


# ---------------------------------------------------------------------------
# Bounceback BCs
# ---------------------------------------------------------------------------

def test_bounceback_bc_horizontal_stationary():
    """
    Stationary horizontal bounceback wall at row 0 (axis=0 boundary).
    The horizontal_wall pair (d_i=4, u_i=2) maps f_4 -> f_2 with no wall term.
    """
    g = _make_grid(nx=6, ny=6)
    g.bc_vertical = ['periodic', 'periodic']
    g.bc_horizontal = ['bounceback', 'periodic']

    val = 0.5
    g.grid[0, :, 4] = val  # f_4 (cx=0, cy=-1) at the wall row

    do_stream(g)

    np.testing.assert_allclose(
        g.grid[0, :, 2], val, rtol=1e-12,
        err_msg="f_4 was not reflected as f_2 by the stationary horizontal wall"
    )


def test_bounceback_bc_vertical_stationary():
    """
    Stationary vertical bounceback wall at column 0 (axis=1 boundary).
    The vertical_wall pair (l_i=3, r_i=1) maps f_3 -> f_1 with no wall term.
    """
    g = _make_grid(nx=6, ny=6)
    g.bc_vertical = ['bounceback', 'periodic']
    g.bc_horizontal = ['periodic', 'periodic']

    val = 0.5
    g.grid[:, 0, 3] = val  # f_3 (cx=-1, cy=0) at the left wall column

    do_stream(g)

    np.testing.assert_allclose(
        g.grid[:, 0, 1], val, rtol=1e-12,
        err_msg="f_3 was not reflected as f_1 by the stationary vertical wall"
    )


def test_bounceback_bc_horizontal_moving():
    """
    Moving horizontal wall at row 0 with x-velocity U_w.
    Starting from all-zero distributions the reflected values must equal the
    wall-correction term exactly:

        wall_term = -2 * cs2_inv * w[d_i] * rho_w * (cx[d_i]*U_w + cy[d_i]*0)

    Checked for the two diagonal pairs that have a non-zero x-component:
      (d_i=7, u_i=5): cx[7]=-1  ->  wall_term = +rho_w * U_w / 6
      (d_i=8, u_i=6): cx[8]=+1  ->  wall_term = -rho_w * U_w / 6
    And for the straight pair where cx[4]=0 -> wall_term=0:
      (d_i=4, u_i=2): wall_term = 0
    """
    rho_0 = 1.0
    U_w = 0.01

    g = _make_grid(nx=6, ny=6)
    g.rho[:] = rho_0
    g.bc_vertical = ['periodic', 'periodic']
    g.bc_horizontal = ['bounceback', 'periodic']
    g.bc_horizontal_kwarg = {'Uwall': [[U_w, 0.], [0., 0.]]}

    # all distributions start at zero; rho_w = mean(rho) = rho_0
    do_stream(g)

    # w[7] = w[8] = 1/36, so 6 * w[7] * rho_0 = rho_0/6
    expected_f5 =  rho_0 * U_w / 6.
    expected_f6 = -rho_0 * U_w / 6.

    np.testing.assert_allclose(g.grid[0, :, 5], expected_f5, rtol=1e-12,
                               err_msg="f_5 wall correction incorrect")
    np.testing.assert_allclose(g.grid[0, :, 6], expected_f6, rtol=1e-12,
                               err_msg="f_6 wall correction incorrect")
    np.testing.assert_allclose(g.grid[0, :, 2], 0., atol=1e-15,
                               err_msg="f_2 should have no wall correction (cx[4]=0)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
