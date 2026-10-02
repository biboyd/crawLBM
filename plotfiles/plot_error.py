import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

from calc_minmax import calc_minmax


def main(infiles, outdir, minmax_list=None):

    time_fig, (mag_ax, time_ax) = plt.subplots(1, 2, figsize=(6.4*2, 4.6))
    time_ax.set_ylabel(f'rms of U_i / |U_i|')
    time_ax.set_xlabel('time')

    mag_ax.set_ylabel(f'mean |U|/u_0')
    mag_ax.set_xlabel('time')
    for f in infiles:
        # extract base file name
        basefile = f.split('/')[-1]
        basename = basefile.removesuffix('.npy')
        analytic_name = f"analytic{basename.removeprefix('plt')}"


        # recover time from basename
        t = int(basename.removeprefix('plt'))
        curr_vel = np.load(f)[:, :, -2:]
        analytic_vel = np.load(analytic_name+'.npy')

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        var_names = ["u_x", "u_y"]

        # loop over do u_x and u_y
        for i, name in enumerate(var_names):

            vel = curr_vel[:, :, i]
            ana_vel = analytic_vel[:, :, i]

            # calc rel diff |delU/Uanalytic|
            u_0 = 0.01
            rel_diff  = np.abs(vel - ana_vel)/u_0

            # calc rms
            rms = np.sqrt(np.mean((vel - ana_vel)**2))/np.sqrt(np.mean(vel**2))
            #rms = np.sqrt(np.mean((vel )**2))/u_0

            lmag = axes[0, i].imshow(vel/u_0, cmap='RdBu', vmin=-1, vmax=1, origin='lower') 

            max_diff = np.max(rel_diff)
            ldiff = axes[1, i].imshow(rel_diff, origin='lower')#, vmin=-max_diff, vmax=max_diff) 

            fig.colorbar(lmag)
            fig.colorbar(ldiff)
            axes[0, i].set_title(f"{name} / $U_0$" )
            axes[1, i].set_title(f"$\\Delta$ {name} / $U_0$" )


            if name == 'u_x':
                tcolor = 'tab:blue'
                tmark = 'o'
            else:
                tcolor = 'tab:orange'
                tmark = 'X'
            if t == 0:
                time_ax.plot(t, rms, tmark, color=tcolor, label=f'rms {name}')
            else:
                time_ax.plot(t, rms, tmark, color=tcolor)
                print(f'Time: {t:0.2e}; Err: {rms:0.4e}')

        mean_mag = np.mean( np.sqrt(analytic_vel[:, :, 0]**2 + analytic_vel[:, :, 1]**2)   )

        mag_ax.plot(t, mean_mag/u_0, 'ko', ) 
        # save file
        fig.tight_layout()
        fig.savefig(f"{outdir}/analytic_comp{basename}.png", bbox_inches='tight')

        # close the fig
        plt.close(fig)

    # save time figure
    time_ax.legend()
    time_fig.tight_layout()
    time_fig.savefig(f"{outdir}/error_over_time.png")
    
            




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