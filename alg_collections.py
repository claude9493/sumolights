# usage:
# import alg_collections
# if tsc in alg_collections.tsc_rl: ......

# For traffic signal controllers
tsc_traditional = ['websters', 'maxpressure', 'sotl', 'uniform']
tsc_rl = ['dqn', 'dqn_queue', 'dqn_pressure', 'doubledqn', 'doubledqn_pressure', 'ddpg', 'td3']

# For neural networks
nn_dqn = ['dqn', 'dqn_queue', 'dqn_pressure', 'doubledqn', 'doubledqn_pressure']
nn_ddpg = ['ddpg']
nn_td3 = ['td3']

# For RL agents
rl_dqn = ['dqn', 'dqn_queue', 'dqn_pressure']
rl_doubledqn = ['doubledqn', 'doubledqn_pressure']
rl_ddpg = ['ddpg']
rl_td3 = ['td3']
