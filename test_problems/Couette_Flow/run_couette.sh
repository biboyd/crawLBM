#!/bin/bash

#paths to relevant scripts
sim_path=./
plot_path=../../plotfiles

# set variables
problem='Couette'
Lx=0.05
Ly=0.1
res=100
U_wall=0.001
tau=0.8
steps=4e4
plots=1000

# Run the sim
python ${sim_path}/init_sim.py --init ${problem} --domain_size $Lx $Ly   \
                               --domain_resolution $res --U_ref $U_wall  \
                               --tau $tau --max_step=$steps              \
                               --plot_int=$plots --plot_dir plotfiles/   \
                               --init_steps=0 

# generate plots
mkdir -p images/ analytic_images/
python ${plot_path}/plot_plotfile.py plotfiles/plt*.npy plt_after_initialization.npy -o images/

# calc error vs analytic soln
python plot_couette_error.py plotfiles/plt00*.npy -o analytic_images/
