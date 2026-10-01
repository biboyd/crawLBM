#!/bin/bash

#paths to relevant scripts
sim_path=../../src/
plot_path=../../plotfiles

# set variables
Nx=50
Ny=50
tau=1.2
steps=1e3
plots=100

# Run the sim
python ${sim_path}/init.py --domain_size $Nx $Ny         \
                      --tau $tau --max_step=$steps     \
                      --plot_int=$plots --init_steps=0 

# generate plots
python ${plot_path}/plot_plotfile.py plt*.npy plt_after_initialization.npy 

# calc error vs analytic soln
python ${plot_path}/plot_error.py plt00*.npy
