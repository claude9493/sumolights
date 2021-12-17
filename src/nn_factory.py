import os

import tensorflow as tf
import alg_collections
from src.neuralnets.dqn import DQN
from src.neuralnets.ddpgactor import DDPGActor
from src.neuralnets.ddpgcritic import DDPGCritic
from src.neuralnets.td3critic import TD3Critic
from src.neuralnets.td3actor import TD3Actor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def nn_factory( nntype, input_d, output_d, args, learner, load, tsc, n_hidden, sess=None):
    nn = None
    hidden_layers = [input_d*n_hidden, input_d*n_hidden]

    if nntype in alg_collections.nn_dqn:
         nn = DQN(input_d, hidden_layers,     
                  args.hidden_act, output_d,  
                  'linear', args.lr,          
                  args.lre, learner=learner)  
    elif nntype in alg_collections.nn_ddpg:
        nn = {}
        nn['actor'] = DDPGActor(input_d, hidden_layers,     
                                args.hidden_act, output_d,  
                                'tanh', args.lr, args.lre,  
                                args.tau, learner=learner,  
                                name='actor'+tsc,           
                                batch_size=args.batch,
                                sess=sess)      
        if learner:
            #only need ddpg critic on learner procs
            nn['critic'] = DDPGCritic(input_d, hidden_layers,  
                                      args.hidden_act, 1,      
                                      'linear', args.lrc,      
                                      args.lre, args.tau,      
                                      learner=learner,         
                                      name='critic'+tsc,
                                      sess=sess)
    elif nntype in alg_collections.nn_td3:
        nn = {}
        nn['actor'] = TD3Actor(input_d, hidden_layers,
                                args.hidden_act, output_d,
                                'tanh', args.lr, args.lre,
                                args.tau, learner=learner,
                                name='actor'+tsc,
                                batch_size=args.batch,
                                sess=sess)
        if learner:
            #only need ddpg critic on learner procs
            nn['critic_1'] = TD3Critic(input_d, hidden_layers,
                                      args.hidden_act, 1,
                                      'linear', args.lrc,
                                      args.lre, args.tau,
                                      learner=learner,
                                      name='critic_1'+tsc,
                                      sess=sess)
            nn['critic_2'] = TD3Critic(input_d, hidden_layers,
                                      args.hidden_act, 1,
                                      'linear', args.lrc,
                                      args.lre, args.tau,
                                      learner=learner,
                                      name='critic_2' + tsc,
                                      sess=sess)
    else:
        #raise not found exceptions
        assert 0, 'Supplied traffic signal control argument type '+str(tsc)+' does not exist.'

    return nn

def get_in_out_d(tsctype, n_incoming_lanes, n_phases):
    #+1 for the all red phase (i.e., terminal state, no vehicles at intersection)
    input_d = (n_incoming_lanes*2) + n_phases + 1
    if tsctype in alg_collections.nn_dqn:
        return input_d, n_phases
    elif tsctype in alg_collections.nn_ddpg:
        return input_d, 1
    elif tsctype in alg_collections.nn_td3:
        return input_d, 1
    else:
        #raise not found exceptions
        assert 0, 'Supplied traffic signal control argument type '+str(tsctype)+' does not exist.'

def gen_neural_networks(args, netdata, tsctype, tsc_ids, learner, load, n_hidden):
        neural_nets = {}
        if tsctype in alg_collections.tsc_rl:
            sess = None
            #if using tf, prepare necessary
            if tsctype in alg_collections.nn_ddpg or tsctype in alg_collections.nn_td3:

                #config = tf.ConfigProto(intra_op_parallelism_threads=1, 
                #                        inter_op_parallelism_threads=1, 
                #                        allow_soft_placement=True)

                tf.compat.v1.reset_default_graph()
                sess = tf.compat.v1.Session()
                #sess = tf.compat.v1.Session(config=config)

            #get desired neural net for each traffic signal controller
            for tsc in tsc_ids:
                input_d, output_d = get_in_out_d(tsctype,
                                                 len(netdata['inter'][tsc]['incoming_lanes']),
                                                 len(netdata['inter'][tsc]['green_phases']))

                neural_nets[tsc] = nn_factory(tsctype, 
                                              input_d, 
                                              output_d, 
                                              args, 
                                              learner, 
                                              load, 
                                              tsc,
                                              n_hidden,
                                              sess=sess)
 
            #if using tf, init all vars
            if tsctype in alg_collections.nn_ddpg or tsctype in alg_collections.nn_td3:
                sess.run(tf.compat.v1.global_variables_initializer())

            #load the saved weights
            if load:                                                
                print('Trying to load '+str(tsctype)+' parameters ...')
                path_dirs = [args.save_path, args.tsc]
                for tsc in tsc_ids:                                    
                    if tsctype in alg_collections.nn_dqn:
                        path = '/'.join(path_dirs+[tsc])               
                        neural_nets[tsc].load_weights(path)        
                    elif tsctype in alg_collections.nn_ddpg:
                        for n in neural_nets[tsc]:                     
                            path = '/'.join(path_dirs+[n,tsc])         
                            neural_nets[tsc][n].load_weights(path)
                    elif tsctype in alg_collections.nn_td3:
                        for n in neural_nets[tsc]:
                            path = '/'.join(path_dirs+[n,tsc])
                            neural_nets[tsc][n].load_weights(path)

                print('... successfully loaded '+str(tsctype)+' parameters')
        return neural_nets
