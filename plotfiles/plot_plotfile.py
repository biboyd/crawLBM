import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

from calc_minmax import calc_minmax

from matplotlib.colors import LogNorm

def main(infiles, outdir, minmax_list=None):

    for f in infiles:
        # extract base file name
        basefile = f.split('/')[-1]
        basename = f.removesuffix('.npy')

        curr_arr = np.load(f)

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        var_names = ["rho", "p", "u_x", "u_y"]

        # decide whether to use minmax
        if minmax_list is None:
            var_range = [[None, None], 
                         [None, None],
                         [None, None],
                         [None, None],
                        ]
        else:
            var_range = minmax_list

        # set grid for streamlines
        X, Y = np.meshgrid(np.arange(0, len(curr_arr[:, 0, 0])), 
                           np.arange(0, len(curr_arr[0, :, 0])))
        # loop over all variables
        for i, (name, ax, curr_range) in enumerate(zip(var_names, axes.flatten(), var_range)):

            var = curr_arr[:, :, i]
            # plot heat map
            if name == "p":
                norm='log'
            elif name == "u_x":
                norm='symlog'
            else:
                norm=None
            curr_im = ax.imshow(var, vmin=curr_range[0], vmax=curr_range[1], norm=norm, origin='lower') 
            fig.colorbar(curr_im)

            # plot streamlines
            if name == 'rho':
                ax.streamplot(X, Y, curr_arr[:, :, 2], curr_arr[:, :, 3], color='k')

            ax.set_title(name)
        
        # save file
        fig.tight_layout()
        fig.savefig(f"{outdir}/{basename}.png", bbox_inches='tight')

        # close the fig
        plt.close(fig)
            

def analytic_soln(Nsteps=1e3, U_0=0.1, k=2*np.pi, tc=1):
    # create x-y mesh
    x_axis = np.linspace(0, 1, 100)
    y_axis = np.linspace(0, 1, 100)
    x_arr, y_arr = np.meshgrid(x_axis, y_axis)

    # set init velocity
    def ux(x_arr, y_arr, t):
        return -U_0 * np.cos(k * x_arr) * np.sin(k * y_arr) * np.exp(-t/tc)

    def uy(x_arr, y_arr, t):
        return U_0 * np.sin(k * x_arr) * np.cos(k * y_arr) * np.exp(-t/tc)

    for i in range(int(Nsteps)):
        Ux = ux(x_arr, y_arr, i)
        Uy = uy(x_arr, y_arr, i)

        
        fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(8, 8))

        curr_imx = ax_x.imshow(Ux, origin='lower')
        fig.colorbar(curr_imx)
        curr_imy = ax_y.imshow(Uy, origin='lower')
        fig.colorbar(curr_imy)

        fig.savefig(f"analytic_pfile{i:07d}.png")
        plt.close(fig)



if __name__ == '__main__':

    parser = ArgumentParser(
                        prog='CrawLBM',
                        description='Runs CrawLBM which will (very slowly) \
                                     evolve a fluid field using a LB method',
                        )

    #parser.add_argument('--inputs', help='inputs file for the arg')

    parser.add_argument('input_files', nargs='*', type=str, 
                        help='files to plot out')

    parser.add_argument('-o', '--outdir', nargs=1, type=str, 
                        default='./',
                        help='directory to plot out to')
    args = parser.parse_args()

    minmax_list = calc_minmax('./', 'plt00')
    main(args.input_files, args.outdir, minmax_list)

    #analytic_soln(Nsteps=1e3, U_0=0.1, k=2*np.pi, tc=1)