import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

def main(infiles, outdir):

    for f in infiles:
        # extract base file name
        basefile = f.split('/')[-1]
        basename = f.removesuffix('.npy')

        curr_arr = np.load(f)
        print(curr_arr.shape)

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        var_names = ["rho", "p", "u_x", "u_y"]

        # loop over all variables
        for i, (name, ax) in enumerate(zip(var_names, axes.flatten())):

            var = curr_arr[:, :, i]
            # plot heat map
            ax.imshow(var) 

            ax.set_title(name)
        
        # save file
        fig.tight_layout()
        fig.savefig(f"{outdir}/{basename}.png", bbox_inches='tight')

        # close the fig
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

    main(args.input_files, args.outdir)
