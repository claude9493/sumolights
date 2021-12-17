#!/usr/bin/env bash
#python run.py -sim single -tsc td3 -lr 0.0005 -lrc 0.0001 -lre 0.000001 -nreplay 5000 -nsteps 2 -target_freq 128 -updates 5000 -batch 32 -save -nogui -n_hidden 3 -mode train -save_t 30 -tau 0.01

# A simpler version training
python run.py -n 1 -sim single -tsc td3 -lr 0.0005 -lrc 0.0001 -lre 0.000001 -nreplay 5000 -nsteps 2 -target_freq 128 -updates 50 -batch 32 -save -nogui -n_hidden 3 -mode train -save_t 30 -tau 0.01


