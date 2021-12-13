from src.rlagents.dqnagent import DQNAgent
from src.rlagents.ddpgagent import DDPGAgent
from src.rlagents.doubledqnagent import DoubleDQNAgent
import alg_collections

def rl_factory(rl_type, args, neural_network, exp_replay, rl_stats, n_actions, eps):
    if rl_type in alg_collections.rl_dqn:
        return DQNAgent(neural_network,
                        eps,                                     
                        exp_replay,                                   
                        n_actions,                                    
                        args.nsteps,                                  
                        args.batch,                                   
                        args.nreplay,                                 
                        args.gamma,                                   
                        rl_stats,
                        args.mode,
                        args.updates)
    elif rl_type in alg_collections.rl_ddpg:
        return DDPGAgent(neural_network,
                         eps,     
                         exp_replay,              
                         n_actions,                
                         args.nsteps,              
                         args.batch,               
                         args.nreplay,             
                         args.gamma,               
                         rl_stats,                
                         args.mode,
                         args.updates)
    elif rl_type in alg_collections.rl_doubledqn:
        return DoubleDQNAgent(neural_network,
                        eps,
                        exp_replay,
                        n_actions,
                        args.nsteps,
                        args.batch,
                        args.nreplay,
                        args.gamma,
                        rl_stats,
                        args.mode,
                        args.updates)

    else:
        #raise not found exceptions
        assert 0, 'Supplied rl argument type '+str(rl_type)+' does not exist.'
