import constants
import numpy as np
import dill
from tensorflow import keras
from keras.optimizers import adam_v2
from keras.models import load_model

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), 
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def remember(self, state, state_, action, reward, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

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
                mem_size=1000000):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.save_path = constants.SAVE_PATH
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