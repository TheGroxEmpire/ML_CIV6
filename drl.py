import dill
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import keras.backend as K
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


class BasePPO(object):

    def __init__(self, action_space, observation_space,scope, args):
        self.scope = scope
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_bound = [self.action_space.low, self.action_space.high]
        self.num_state = self.observation_space.shape[0]
        self.num_action = self.action_space.shape[0]
        self.cliprange = args.cliprange
        self.checkpoint_path = args.checkpoint_dir+'/'+args.environment + '/' + args.policy
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.environment = args.environment
        with tf.variable_scope('input'):
            self.s = tf.placeholder("float", [None, self.num_state])
        with tf.variable_scope('action'):
            self.a = tf.placeholder(shape=[None, self.num_action], dtype=tf.float32)
        with tf.variable_scope('target_value'):
            self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        with tf.variable_scope('advantages'):
            self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    def build_critic_net(self, scope):

        raise NotImplementedError("You can't instantiate this class!")

    def build_actor_net(self, scope, trainable):

        raise NotImplementedError("You can't instantiate this class!")


    def build_net(self):

        self.value  = self.build_critic_net('value_net')
        pi, pi_param = self.build_actor_net('actor_net', trainable= True)
        old_pi, old_pi_param = self.build_actor_net('old_actor_net', trainable=False)
        self.syn_old_pi = [oldp.assign(p) for p, oldp in zip(pi_param, old_pi_param)]
        self.sample_op = tf.clip_by_value(tf.squeeze(pi.sample(1), axis=0), self.action_bound[0], self.action_bound[1])[0]


        with tf.variable_scope('critic_loss'):
            self.adv = self.y - self.value
            self.critic_loss = tf.reduce_mean(tf.square(self.adv))

        with tf.variable_scope('actor_loss'):
            ratio = pi.prob(self.a) / old_pi.prob(self.a)   #(old_pi.prob(self.a)+ 1e-5)
            pg_losses= self.advantage * ratio
            pg_losses2 = self.advantage * tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            self.actor_loss = -tf.reduce_mean(tf.minimum(pg_losses, pg_losses2))

    def load_model(self, sess, saver):
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('.............Model restored to global.............')
        else:
            print('................No model is found.................')

    def save_model(self, sess, saver, time_step):
        print('............save model ............')
        saver.save(sess, self.checkpoint_path + '/'+self.environment +'-' + str(time_step) + '.ckpt')

    def choose_action(self, s, sess):
        s = s[np.newaxis, :]
        a = sess.run(self.sample_op, {self.s: s})
        return a

    def get_v(self, s, sess):
        if s.ndim < 2: s = s[np.newaxis, :]
        return sess.run(self.value, {self.s: s})[0, 0]