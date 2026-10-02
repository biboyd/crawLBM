#!/bin/bash

#paths to relevant scripts
sim_path=../../src/
plot_path=../../plotfiles

# set variables
problem='Couette'
Lx=0.1
Ly=0.05
res=100
U_wall=0.001
tau=0.8
steps=4e3
plots=100

# Run the sim
python ${sim_path}/init.py --init ${problem} --domain_size $Lx $Ly         \
                        --domain_resolution $res --U_ref $U_wall \
                      --tau $tau --max_step=$steps                         \
                      --plot_int=$plots --plot_dir plotfiles/              \
                      --init_steps=0 

# generate plots
#mkdir -p images/ analytic_images/
#python ${plot_path}/plot_plotfile.py plotfiles/plt*.npy plt_after_initialization.npy -o images/
#
## calc error vs analytic soln
#python plot_couette_error.py plotfiles/plt00*.npy -o analytic_images/
