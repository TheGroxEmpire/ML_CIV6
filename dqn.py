import constants
import numpy as np
import dill
import random
from tensorflow import keras
from keras.optimizers import adam_v2
from keras.models import load_model

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
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

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

class ReplayBufferPER():
    def __init__(self, max_size, input_shape, alpha):
        self.max_size = max_size
        self.alpha = alpha
        self.priority_sum = [0 for _ in range(2 * self.max_size)]
        self.priority_min = [float('inf') for _ in range(2 * self.max_size)]
        self.max_priority = 1.

        self.data = {
            'states': np.zeros(shape=(max_size, *input_shape), dtype=np.uint8),
            'actions': np.zeros(shape=max_size, dtype=np.int32),
            'rewards': np.zeros(shape=max_size, dtype=np.float32),
            'new_states': np.zeros(shape=(max_size, *input_shape), dtype=np.uint8),
            'done': np.zeros(shape=max_size, dtype=bool)
        }

        self.mem_cntr = 0
        self.size = 0

    def remember(self, state, state_, action, reward, done):
        idx = self.mem_cntr
        self.data['states'][idx] = state
        self.data['actions'][idx] = action
        self.data['rewards'][idx] = reward
        self.data['new_states'][idx] = state_
        self.data['done'][idx] = done

        self.mem_cntr = (idx + 1) % self.max_size

        self.size = min(self.max_size, self.size + 1)

        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)
    
    def _set_priority_min(self, idx, priority_alpha):
        idx += self.max_size
        self.priority_min[idx] = priority_alpha

        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        idx += self.max_size
        self.priority_sum[idx] = priority

        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]
    
    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.max_size:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1
            
        return idx - self.max_size

    def sample_buffer(self, batch_size, beta):
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx
        
        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]

        prob = self.priority_sum[idx + self.max_size] / self._sum()
        weight = (prob * self.size) ** (-beta)
        samples['weights'][i] = weight / max_weight

        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, samples['weights'], replace=False)
        index = samples['indexes'][batch]
        states = samples['states'][batch]
        actions = samples['actions'][batch]
        rewards = samples['rewards'][batch]
        states_ = samples['new_states'][batch]
        dones = samples['dones'][batch]
        return index, states, actions, rewards, states_, dones

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)
    
    def is_full(self):
        return self.max_size == self.size

def build_dqn(lr, n_actions, fc1_dims, fc2_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)])
    model.compile(optimizer=adam_v2.Adam(learning_rate=lr), loss='mean_squared_error')

    return model

class Agent():
    def __init__(self, state, 
                n_actions, 
                lr=0.0005, 
                gamma=0.99, 
                epsilon=1.0, 
                batch_size=64,
                eps_dec=1e-3, 
                eps_min=0.01,
                mem_size=100000,
                PER=True,
                alpha=0.6,
                beta=0.4):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.PER = PER
        self.alpha = alpha
        self.beta = beta
        self.save_path = constants.SAVE_PATH
        if self.PER:
            self.memory = ReplayBufferPER(mem_size, state.shape, self.alpha)
        else:
            self.memory = ReplayBuffer(mem_size, state.shape)
            
        self.q_eval = build_dqn(lr, n_actions, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.remember(state, action, reward, new_state, done)

    def act(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)

        return action

    def replay(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        if self.PER:
            indexes, states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)
        else:
            states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)


        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next, axis=1)*dones


        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min

    def save_model(self, file_name):
        print("Saving model, do not terminate the program")
        self.q_eval.save(f"{self.save_path}{file_name}")
        with open(f"{self.save_path}{file_name}/parameters.pkl", 'wb') as f:
            dill.dump([self.gamma, self.epsilon, self.eps_dec, self.eps_min, self.batch_size, self.memory], f)

        print("Model saved")

    def load_model(self, file_name):
        self.q_eval = load_model(f"{self.save_path}{file_name}")
        with open(f"{self.save_path}{file_name}/parameters.pkl", 'rb') as f:
            self.gamma, self.epsilon, self.eps_dec, self.eps_min, self.batch_size, self.memory = dill.load(f)
        
        print("Model loaded")