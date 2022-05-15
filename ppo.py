import constants
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras.losses import MSE
from keras.optimizers import adam_v2

class Agent:
    def __init__(self, 
                state, 
                action_size, 
                gamma=0.99, 
                alpha=0.0003,
                gae_lambda=0.95, 
                policy_clip=0.2, 
                batch_size=64, 
                n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.save_path = constants.SAVE_PATH
        self.action_size = action_size
        self.n_epochs = n_epochs

        self.actor = ActorNetwork(action_size)
        self.actor.compile(optimizer=adam_v2.Adam(learning_rate=alpha))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=adam_v2.Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)

        self.log_prob = []
        self.value = []

    def remember(self, state, next_state, action, reward, done):
        self.memory.store_memory(state, action, self.log_prob, self.value, reward, done)

    def save_model(self, file_name):
        try:
            print("Saving model, do not terminate the program")
            self.actor.save(f"{self.save_path}{file_name}/actor")
            self.critic.save(f"{self.save_path}{file_name}/critic")
            print("Model saved")
        except:
            print("Failed to save model")
    def load_models(self, file_name):
        try:
            self.actor = load_model(f"{self.save_path}{file_name}/actor")
            self.critic = load_model(f"{self.save_path}{file_name}/critic")
            print("Model loaded")
        except:
            print("Model can't be loaded")

    def act(self, observation):
        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        self.log_prob = dist.log_prob(action)
        self.value = self.critic(state)

        action = action.numpy()[0]
        self.value = self.value.numpy()[0]
        self.log_prob = self.log_prob.numpy()[0]

        return action

    def replay(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (
                        1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda

                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                        1-self.policy_clip,
                                                        1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                    weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(
                    #                                  returns-critic_value, 2))
                    critic_loss = MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))

        self.memory.clear_memory()

class ActorNetwork(Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class CriticNetwork(Model):
    def __init__(self, fc1_dims=512, fc2_dims=512):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q = self.q(x)

        return q

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []