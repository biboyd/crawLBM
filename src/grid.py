"""
class for holding the grid itself.
thinking it'll just be a bunch of numpy
arrays or maybe just one big one?
"""

import numpy as np


class Grid:
    """
    Class to contain the grid info and
    particularly the distribution functions.

    initialize w/ domain size nx, ny

    use Grid.grid to return the numpy array (size (nx, ny, 9))

    Distribution functions numbered as:
    6     2     5
      .   .   .
        . . .
    3 . . 0 . . 1
        . . .
      .   .   .
    7     4     8

    0 is centered - zero velocity
    1-4 are unit one velocities (horizontal/vertical)
    5-8 are diagonals sqrt(2) velocities
    
    """
    def __init__(self, nx, ny):
        # set dimensions of grid
        self.nx = nx
        self.ny = ny

        # set num of distributions
        self.nf = 9
        
        self.cardinal = np.arange(1, 5, 1)
        self.diag = np.arange(5, 9, 1)

        # setup actual weights and vel and such
        self.weights = np.array([4./9, 1./9., 1./9., 1./9., 1./9.,
                                 1./36., 1./36., 1./36.,1./36., ])
        self.cx = np.array([0, 1., 0, -1., 0, 1., -1., -1., 1.])
        self.cy = np.array([0, 0, 1., 0, -1., 1., 1., -1., -1.])

        # init the grid
        self.grid = np.empty((self.nx, self.ny, self.nf))

        # macro vars
        self.rho = np.empty((self.nx, self.ny))
        self.uvec = np.empty((self.nx, self.ny, 2))

