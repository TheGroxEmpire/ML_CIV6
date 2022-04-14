import dill
import numpy as np
import keras.backend as K
from keras.metrics import MeanSquaredError
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Flatten, Add, Subtract, Lambda
import random
from collections import deque

class Vanilla_DQN():
    def __init__(self,
                state,
                action_size,
                epsilon=1, 
                epsilon_decay=0.995, 
                epsilon_min=0.01, 
                batch_size=32, 
                discount_factor=0.9):
        self.state_size = state.shape[1]
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.save_path = "./saved_model/"
        self.memory = deque(maxlen=100000)
        self.model = self.create_model()
    
    def create_model(self):
        """Create vanilla DQN model"""
        input = Input(shape=(self.state_size))

        out = Dense(512, activation='relu')(input)
        out = Dense(512, activation='relu')(out)
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
        return np.argmax(self.model.predict(state))

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
                target = reward + self.discount_factor * np.max(self.model.predict(next_state))
            final_target = self.model.predict(state)
            final_target[0][action] = target
            self.model.fit(state, final_target, verbose=0)
    
    def save_model(self, file_name):
        try:
            print("Saving model, do not terminate the program")
            self.model.save(f"{self.save_path}{file_name}")
            with open(f"{self.save_path}{file_name}\parameters.pkl", 'wb') as f:
                dill.dump([self.epsilon, self.epsilon_decay, self.epsilon_min,
                self.discount_factor, self.memory, self.batch_size], f)
            print("Model saved")
        except:
            print("Failed to save model")


    def load_model(self, file_name):
        try:
            self.model = load_model(f"{self.save_path}{file_name}")
            with open(f"{self.save_path}{file_name}\parameters.pkl", 'rb') as f:
                self.epsilon, self.epsilon_decay, self.epsilon_min, self.discount_factor, self.memory, self.batch_size = dill.load(f)
            print("Loading model")
        except:
            print("Model can't be loaded")

       

class Dueling_DQN(Vanilla_DQN):
    def __init__(self,
                state,
                action_size,
                epsilon=1, 
                epsilon_decay=0.995, 
                epsilon_min=0.01, 
                batch_size=32, 
                discount_factor=0.9):
        
        super().__init__(state,
                action_size,
                epsilon, 
                epsilon_decay, 
                epsilon_min, 
                batch_size, 
                discount_factor)
    
    def create_model(self):
        """Create Dueling DQN model"""
        input = Input(shape=(self.state_size))

        value = Dense(512, activation="relu")(input)
        value = Dense(1, activation="relu")(value)
        advantage = Dense(512, activation="relu")(input)
        advantage = Dense(self.action_size, activation="relu")(advantage)
        advantage_mean = Lambda(lambda x: K.mean(x, axis=1))(advantage)
        advantage = Subtract()([advantage, advantage_mean])
        out = Add()([value, advantage])

        model = Model(inputs=input, outputs=out)
        model.compile(optimizer="adam", loss="mse")

        return model