# run_game.py
# Basic file how to run the game and control it with an AI

import os
import tensorflow as tf
import pettingzoo_env
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import ppo
import dqn
import dueling_ddqn

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv

if __name__ == '__main__':
    # --- Load / save setting
    enable_load = False
    enable_save = False
    
    # --- Set up your algorithm here
    N_EPISODES = 1
    N_TURNS = 20
    '''Algorithm list:
        - dqn
        - dueling_ddqn
        - ppo
    '''
    algorithm_version = 'dueling_ddqn'

    comment_suffix = "a(3w2s)-d(2w1s)_default"

    env = pettingzoo_env.PettingZooEnv(N_TURNS, "show")
    def env_creator():
        return env
     
    register_env("env", lambda config: PettingZooEnv(env_creator()))
    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    env.reset()

    # --- Data list for plot
    attacker_r = np.array([])
    defender_r = np.array([])
    episodes = np.array([])

    # --- instantiate agentss
    algorithm_dict = {
        'dqn': dqn.Agent,
        'dueling_ddqn': dueling_ddqn.Agent,
        'ppo': ppo.Agent,
    }
    # Attacker action space
    # attacker_agent = algorithm_dict[algorithm_version](state, 7)
    attacker_agent = PPOTrainer(config={
                                "env": "env",
                                })
    # Defender action space
    defender_agent  = PPOTrainer(config={
                                "env": "env",
                                })

    agent_dict = {
        'attacker' : attacker_agent,
        'defender' : defender_agent
    }

    if enable_save:
        if not os.path.exists(f"./logs/{algorithm_version}_{comment_suffix}"):
            os.makedirs(f"./logs/{algorithm_version}_{comment_suffix}")
        tensorboard_writer = tf.summary.create_file_writer(logdir=f"./logs/{algorithm_version}_{comment_suffix}")

    # --- load saved model
    if enable_load:
        with open(f"./plots/{algorithm_version}_{comment_suffix}.csv") as f:
            lines = list(csv.reader(f))
        lines = np.array(lines, float)
        attacker_r, defender_r = lines[:2]
        # Cumulative episodes is the number of episode from previous load. If there's nothing to load, start at 0
        cumulative_episodes = len(attacker_r)
        attacker_agent.load_model(f'attacker_{algorithm_version}_{comment_suffix}')
        defender_agent.load_model(f'defender_{algorithm_version}_{comment_suffix}')
        print("Continuing from last save data")
       
    else:
        cumulative_episodes = 0 

    training_start_time = time.time()
    episode_start = cumulative_episodes+1
    episode_end = cumulative_episodes+N_EPISODES+1
    for epoch in range(episode_start, episode_end):
        # --- Initialize the game by putting units and city on the playing field, etc.
        env.reset()
        # --- Get start time
        s = time.time()
        for agent in env.agent_iter():
            #print(f"{agent} turn")
            policy = agent_dict[agent].get_policy(agent)
            state, reward, done, info = env.last()
            print(state)
            action = agent_dict[agent].compute_single_action(state)
            if agent == 'attacker':
                attacker_reward_episode = reward
            else:
                defender_reward_episode = reward

            if done:
                break

            env.step(action)

        # --- Replay the agent past experience
        attacker_agent.replay()
        defender_agent.replay()
        # --- Store latest reward
        attacker_r = np.append(attacker_r, attacker_reward_episode)
        defender_r = np.append(defender_r, defender_reward_episode) 
        # --- Get end time
        e = time.time()
        print(f"Episode: {epoch}, time spent: {round(e-s, 2)}s")
        # --- Update tensorboard reward
        
        # --- Save model and data value every 100 episodes or at the last episode
        if enable_save and (epoch % 1000 == 0 or epoch == episode_end-1):
            attacker_agent.save_model(f'attacker_{algorithm_version}_{comment_suffix}')
            defender_agent.save_model(f'defender_{algorithm_version}_{comment_suffix}')

            with open(f"./plots/{algorithm_version}_{comment_suffix}.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attacker_r)
                writer.writerow(defender_r)

            with tensorboard_writer.as_default():
                tf.summary.scalar(name=f"attacker_reward", data=np.mean(attacker_r[-1000:]), step=epoch)
                tf.summary.scalar(name=f"defender_reward", data=np.mean(defender_r[-1000:]), step=epoch)
                tensorboard_writer.flush()

    env.close()    
    training_end_time = time.time()
    print(f"Training finished. Total elapsed time: {round(training_end_time-training_start_time, 2)}s")
    # --- Rewards vs episode plot
    def moving_average(x, w=1000):
        avg = np.convolve(x, np.ones(w), 'valid') / w
        avg = np.append(avg, np.repeat(np.nan, w-1))
        return avg

    plt.plot(moving_average(attacker_r), label='Attacker rewards')
    plt.plot(moving_average(defender_r), label='Defender rewards')
    plt.title(f"{algorithm_version}_{comment_suffix}")
    plt.legend() 
    if enable_save:
        plt.savefig(f"plots/rewards_vs_episodes_{algorithm_version}_{comment_suffix}.png")

    plt.show()
    

            





