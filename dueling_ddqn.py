import dill
import tensorflow as tf
import keras
from keras.optimizers import adam_v2
import numpy as np
import dqn

class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A

class Agent(dqn.Agent):
    def __init__(self, state, 
                n_actions, 
                lr=0.0005, 
                gamma=0.99, 
                epsilon=1.0, 
                batch_size=64,
                eps_dec=1e-3, 
                eps_min=0.01, 
                mem_size=10000,
                PER=False,
                alpha=0.6,
                beta=0.4, 
                fc1_dims=256,
                fc2_dims=256, 
                replace=1000):

        super().__init__(state, 
                n_actions, 
                lr, 
                gamma, 
                epsilon, 
                batch_size, 
                eps_dec, 
                eps_min,
                mem_size,
                PER,
                alpha,
                beta)

        self.replace = replace
        self.learn_step_counter = 0
        if self.PER:
            self.memory = dqn.ReplayBufferPER(mem_size, state.shape, self.alpha)
        else:
            self.memory = dqn.ReplayBuffer(mem_size, state.shape)
            
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=adam_v2.Adam(learning_rate=lr),
                            loss='mean_squared_error')
        self.q_next.compile(optimizer=adam_v2.Adam(learning_rate=lr),
                            loss='mean_squared_error')

    def act(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def replay(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        if self.PER:
            indexes, states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size, self.beta)
        else:
            states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)

        for idx, terminal in enumerate(dones):
            #if terminal:
                #q_next[idx] = 0.0
            q_target[idx, actions[idx]] = rewards[idx] + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min

        self.learn_step_counter += 1

    def save_model(self, file_name):
        print("Saving model, do not terminate the program")
        self.q_eval.save_weights(f"{self.save_path}{file_name}")
        self.q_next.save_weights(f"{self.save_path}{file_name}/target_weight")
        with open(f"{self.save_path}{file_name}/parameters.pkl", 'wb') as f:
            dill.dump([self.gamma, self.epsilon, self.eps_dec, self.eps_min, self.batch_size, self.memory], f)
        
        print("Model saved")

    def load_model(self, file_name):
        self.q_eval.load_weights(f"{self.save_path}{file_name}")
        self.q_next.load_weights(f"{self.save_path}{file_name}/target_weight")
        with open(f"{self.save_path}{file_name}/parameters.pkl", 'rb') as f:
            self.gamma, self.epsilon, self.eps_dec, self.eps_min, self.batch_size, self.memory = dill.load(f)
        self.learn_step_counter += 1
        
        print("Model loaded")