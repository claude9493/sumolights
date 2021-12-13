# usage:
# import alg_collections
# if tsc in alg_collections.tsc_rl: ......

# For traffic signal controllers
tsc_traditional = ['websters', 'maxpressure', 'sotl', 'uniform']
tsc_rl = ['dqn', 'dqn_queue', 'dqn_pressure', 'doubledqn', 'doubledqn_pressure', 'ddpg']

# For neural networks
nn_dqn = ['dqn', 'dqn_queue', 'dqn_pressure', 'doubledqn', 'doubledqn_pressure']
nn_ddpg = ['ddpg']

# For RL agents
rl_dqn = ['dqn', 'dqn_queue', 'dqn_pressure']
rl_doubledqn = ['doubledqn', 'doubledqn_pressure']
rl_ddpg = ['ddpg']
