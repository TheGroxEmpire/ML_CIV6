import dill
import tensorflow as tf
import keras
from keras.optimizers import adam_v2
from keras.models import load_model
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


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def remember(self, state, state_, action, reward, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

class Agent(dqn.Agent):
    def __init__(self, state, 
                n_actions, 
                lr=0.0005, 
                gamma=0.99, 
                epsilon=1.0, 
                batch_size=64,
                eps_dec=1e-3, 
                eps_min=0.01, 
                mem_size=100000, 
                fc1_dims=256,
                fc2_dims=256, 
                replace=100):

        super().__init__(state, 
                n_actions, 
                lr, 
                gamma, 
                epsilon, 
                batch_size, 
                eps_dec, 
                eps_min)

        self.replace = replace
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, state.shape)
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=adam_v2.Adam(learning_rate=lr),
                            loss='mean_squared_error')
        # just a formality, won't optimize network
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
        try:
            print("Saving model, do not terminate the program")
            self.q_eval.save(f"{self.save_path}{file_name}")
            with open(f"{self.save_path}{file_name}/parameters.pkl", 'wb') as f:
                dill.dump([self.gamma, self.epsilon, self.eps_dec, self.eps_min, self.batch_size, self.memory], f)

            print("Model saved")
        except:
            print("Failed to save model")

    def load_model(self, file_name):
        try:
            self.q_eval = load_model(f"{self.save_path}{file_name}")
            with open(f"{self.save_path}{file_name}/parameters.pkl", 'rb') as f:
                self.gamma, self.epsilon, self.eps_dec, self.eps_min, self.batch_size, self.memory = dill.load(f)
            print("Model loaded")
        except:
            print("Model can't be loaded")