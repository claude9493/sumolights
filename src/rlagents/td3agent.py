import numpy as np

from src.rlagent import RLAgent

class TD3Agent(RLAgent):
    def __init__(self, networks, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma, rl_stats, mode, updates):
        super().__init__(networks, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma, rl_stats, mode, updates) 
        
    def get_action(self, state):

        if self.mode == 'train':
            self.retrieve_weights('online')
        #forward online actor to get action
        a = self.networks['actor'].forward(state[np.newaxis,...], 'online')
        #add exploration noise
        if len(self.exp_replay) < self.n_exp_replay:
            self.epsilon = 1.0
        a += np.random.uniform(-self.epsilon, self.epsilon, size=self.n_actions)

        #clip action (tanh activation range)
        a[a>1.0] = 1.0
        a[a<-1.0] = -1.0

        ###return continuous action
        return a[0]

    def train_batch(self, update_freq):
        ###sample replay
        sample_batch = self.sample_replay()

        states, actions, targets = self.process_batch(sample_batch)
        targets = np.expand_dims(targets, -1)

        #train critic
        self.networks['critic_1'].backward( states, actions, targets )
        self.networks['critic_2'].backward( states, actions, targets )

        #get grads for actor
        actions = self.networks['actor'].forward(states, 'online')
        grads = self.networks['critic_1'].gradients(states, actions)
        # grads_2 = self.networks['critic_2'].gradients(states, actions)

        #train actor
        # self.networks['actor'].backward(states, min(grads_1[0], grads_2[0]))
        self.networks['actor'].backward(states, grads[0])
        self.rl_stats['updates'] += 1
        self.rl_stats['n_exp'] -= 1
        #send new actor weights for actor processes
        self.send_weights('online')

        #update online with target params periodically
        if self.rl_stats['updates'] % update_freq == 0: 
            self.networks['actor'].transfer_weights()    
            self.networks['critic_1'].transfer_weights()
            self.networks['critic_2'].transfer_weights()

    def process_batch(self, sample_batch):
        #batch compute bootstrap
        next_states, terminals = [], []
        for trajectory in sample_batch:
            ###normalize reward by comparison to maximum reward  
            ###agent has experienced across all actors              
            next_states.append(trajectory[-1]['next_s'])                       
            terminals.append(trajectory[-1]['terminal'])                       

        R = self.next_state_bootstrap(np.stack(next_states), np.stack(terminals))
        #compute targets
        max_r = self.rl_stats['max_r']
        targets, states, actions = [], [], []

        for trajectory, r in zip(sample_batch, R):
            rewards = []
            for exp in trajectory:
                states.append(exp['s'])
                actions.append(exp['a'])
                ###normalize reward by comparison to maximum reward 
                ###agent has experienced across all actors
                rewards.append(exp['r']/max_r)
            targets.extend( self.process_trajectory( rewards, r ) )
                                                                                         
        if self.n_steps > 1:
            idx = np.random.randint(0, len(targets), size = self.n_batch) 
            _states, _actions, _targets = [], [], []
            for i in idx:
                _states.append(states[i])
                _targets.append(targets[i])
                _actions.append(actions[i])
            return np.stack(_states), np.stack(_actions), np.stack(_targets)
        else:
            return np.stack(states), np.stack(actions), np.stack(targets)

    def next_state_bootstrap(self, next_states, terminals):

        bootstrap_actions = self.networks['actor'].forward(next_states, 'target')

        q_next_ind = np.argmax(self.networks['critic_1'].forward(next_states, bootstrap_actions, 'online'), axis=-1)
        q_next_s = self.networks['critic_1'].forward(next_states, bootstrap_actions, 'target')
        R_1 = np.take_along_axis(q_next_s, np.expand_dims(q_next_ind, axis=-1), axis=-1).squeeze(axis=-1)

        q_next_ind = np.argmax(self.networks['critic_2'].forward(next_states, bootstrap_actions, 'online'), axis=-1)
        q_next_s = self.networks['critic_2'].forward(next_states, bootstrap_actions, 'target')
        R_2 = np.take_along_axis(q_next_s, np.expand_dims(q_next_ind, axis=-1), axis=-1).squeeze(axis=-1)

        return [ 0.0 if t is True else r for t, r in zip(terminals, np.minimum(R_1, R_2))]

    def process_trajectory(self, rewards, R):
        #targets = self.compute_targets(rewards, R)
        return self.compute_targets(rewards, R)

    def send_weights(self, nettype):
        self.rl_stats[nettype] = self.networks['actor'].get_weights(nettype)

    def retrieve_weights(self, nettype):
        self.networks['actor'].set_weights(self.rl_stats[nettype], nettype)

    def compute_targets(self, rewards, R):
        ###compute targets using discounted rewards
        target_batch = []

        for i in reversed(range(len(rewards))):
            R = rewards[i] + (self.gamma * R)
            target_batch.append(R)

        target_batch.reverse()
        return target_batch





