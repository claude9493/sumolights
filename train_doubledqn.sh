#!/usr/bin/env bash
python run.py -sim single -tsc doubledqn -lr 0.0005 -lre 0.000001 -nreplay 5000 -nsteps 2 -target_freq 128 -updates 5000 -batch 32 -save -nogui -n_hidden 3 -mode train -gmin 5