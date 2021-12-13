#!/usr/bin/env bash
python run.py -sim single -tsc ddpg -lr 0.0005 -lrc 0.0001 -lre 0.000001 -nreplay 5000 -nsteps 2 -target_freq 128 -updates 5000 -batch 32 -save -nogui -n_hidden 3 -mode train -save_t 30 -tau 0.01



