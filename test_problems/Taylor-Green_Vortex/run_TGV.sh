#!/bin/bash

#paths to relevant scripts
sim_path=./
plot_path=../../plotfiles

# set variables
Lx=0.002
Ly=0.002
nx=64
tau=1
U_ref=0.01
steps=1e3
plots=200

mkdir -p plotfiles/ images/ 
# Run the sim
python ${sim_path}/init_sim.py --domain_size $Lx $Ly        \
                      --domain_resolution $nx         \
                      --tau $tau --U_ref $U_ref         \
                      --max_step=$steps      \
                      --plot_int=$plots --plot_dir plotfiles/ \
                      --init_steps=0 

# generate plots
python ${plot_path}/plot_plotfile.py plotfiles/plt*.npy plt_after_initialization.npy  -o images/

# calc error vs analytic soln
python ${plot_path}/plot_error.py plotfiles/plt00*.npy -o images/
