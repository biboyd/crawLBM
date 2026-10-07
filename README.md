# crawLBM

A bare-bones 2D Lattice Boltzmann Method (LBM) fluid solver written in Python.
The logic is simply: crawl before walking or running.

## Features

* BGK (single-relaxation-time) collision operator
* D2Q9 lattice structure
* Periodic and halfway bounce-back boundary conditions (including moving walls)
* Taylor-Green Vortex and Couette Flow test problems
* Physical unit conversion (SI inputs → lattice units)

## Installation

### From PyPI

```bash
pip install crawlbm
```

### From source

```bash
git clone https://github.com/biboyd/crawLBM.git
cd crawLBM
pip install .
```

For development (editable install):

```bash
pip install -e .
```

## Usage

After installation, run the solver from the command line:

```bash
# Taylor-Green Vortex (default)
crawlbm --init TG-vortex --domain_resolution 64 --max_step 1000 --plot_dir ./output

# Couette Flow
crawlbm --init Couette --domain_resolution 32 --tau 0.8 --max_step 2000 --plot_dir ./output
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--init` | `TG-vortex` | Initialization type: `TG-vortex` or `Couette` |
| `--domain_size Lx Ly` | `1 1` | Physical domain size in SI units |
| `--domain_resolution` | `64` | Grid points along the longest dimension |
| `--tau` | `0.8` | Relaxation time (must be > 0.5) |
| `--nu` | `1e-6` | Physical kinematic viscosity (m²/s) |
| `--U_ref` | `0.01` | Reference velocity scale (m/s) |
| `--max_step` | `1000` | Number of time steps |
| `--plot_int` | `100` | Save a plotfile every N steps |
| `--plot_dir` | `./` | Output directory for plotfiles |

### Python API

```python
from crawlbm import Grid, advance_sim
from crawlbm._cli import init_TG_vortex

grid = Grid(nx=64, ny=64)
init_TG_vortex(grid, Lx=1.0, Ly=1.0, U_0=0.01)
advance_sim(grid, tau=0.8, max_step=500, plot_int=100, plot_dir='./output')
```

## Running tests

```bash
pip install pytest
pytest unit_tests/ -v
```

## Dependencies

* Python >= 3.8
* NumPy

## LLM Acknowledgement
Core functions and functionality were written by Brendan Boyd. Unit tests, convergence tests, and cleaning up the codebase to work as a python package was implemented via Claude Code.

Additionally, some bug fixes were found via Claude Code.
