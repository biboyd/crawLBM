"""
will calc macro variables ie density 
and output them in a file.

will start by just outputting rho, vel in numpy array
"""

def plot_data(grid, outname='pfile'):

    # make sure u and rho are calculated
    calc_rho(grid)
    calc_u(grid)

    rho_u = np.dstack((grid.rho, grid.u_vec))

    np.save(outname, rho_u)