import numpy as np
from os import listdir

def calc_minmax(directory, file_naming):
    rho = [np.inf, -np.inf]
    p = [np.inf, -np.inf]
    Ux = [np.inf, -np.inf]
    Uy = [np.inf, -np.inf]

    minmax_list = [rho, p, Ux, Uy]
    for f in listdir(directory):
        if file_naming in f and f[-4:] == '.npy':
            # load in np array
            try:
                arr = np.load(f"{directory}/{f}")
            except:
                print(f"can't open {f}. skipping it")
                continue
                
            var_names = ["rho", "p", "u_x", "u_y"]

            # calc minmax for each var
            for i, (name, minmax_vals) in enumerate(zip(var_names, minmax_list)):
                minmax_vals[0] = min(minmax_vals[0], np.min(arr[:, :, i]))
                minmax_vals[1] = max(minmax_vals[1], np.max(arr[:, :, i]))

    return minmax_list

if __name__ == "__main__":
    curr_dir = './'
    file_name = 'plt0'

    minmax_list = calc_minmax(curr_dir, file_name)
    var_names = ["rho", "p", "u_x", "u_y"]
    for name, minmax_vals in zip(var_names, minmax_list):
        print(f"{name}:")
        print(f"Min: {minmax_vals[0]:0.2e}")
        print(f"Max: {minmax_vals[1]:0.2e}")