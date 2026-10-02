#!/bin/bash

#paths to relevant scripts
sim_path=../../src/
plot_path=../../plotfiles

# set variables
problem='Couette'
Nx=80
Ny=40
tau=1.2
steps=4e3
plots=100

# Run the sim
python ${sim_path}/init.py --init ${problem} --domain_size $Nx $Ny         \
                      --tau $tau --max_step=$steps                         \
                      --plot_int=$plots --plot_dir plotfiles/              \
                      --init_steps=0 

# generate plots
mkdir -p images/ analytic_images/
python ${plot_path}/plot_plotfile.py plotfiles/plt*.npy plt_after_initialization.npy -o images/

# calc error vs analytic soln
python plot_couette_error.py plotfiles/plt00*.npy -o analytic_images/
