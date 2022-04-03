import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
import random
from collections import deque

class DQN():
    def __init__(self,
                state,
                epsilon=1, 
                epsilon_decay=0.995, 
                epsilon_min=0.01, 
                batch_size=32, 
                discount_factor=0.9):
        self.state_size = state.shape[0]
        self.action_size = 1
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.memory = deque(maxlen=20000)
        self.model = self.create_model()
    
    def create_model(self):
        """Create vanilla DQN model"""
        input = Input(shape=(self.state_size))

        out = Dense(128, activation='relu')(input)
        out = Dense(128, activation='relu')(out)
        out = Dense(self.action_size, activation='linear')(out)

        model = Model(inputs=input, outputs=out)
        model.compile(optimizer="adam", loss="mse")
        return model

    def act(self, state):
        """
        act randomly by probability of epsilon or predict the next move by the neural network model
        :param state:
        :return:
        """
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, next_state, action, reward, done):
        """
        remember the experience
        :param state:
        :param next_state:
        :param action:
        :param reward:
        :param done:
        :return:
        """
        self.memory.append((state, next_state, action, reward, done))

    def replay(self):
        """
        experience replay. find the q-value and train the neural network model with state as input and q-values as targets
        :return:
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        batch = random.choices(self.memory,k=self.batch_size)
        for state, next_state, action, reward, done in batch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.max(self.model.predict(next_state)[0])
            final_target = self.model.predict(state)
            final_target[0][action] = target
            self.model.fit(state,final_target,verbose=0)
    