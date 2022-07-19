# run_game.py
# Basic file how to run the game and control it with an AI

import pygame
import constants
import csv
import time
import numpy as np
import game
import matplotlib.pyplot as plt
import ppo
import dqn
import dueling_ddqn

if __name__ == '__main__':

    # --- Load / save setting
    enable_load = False
    enable_save = True
    
    # --- Set up your algorithm here
    N_EPISODES = 100000
    N_TURNS = 15
    '''Algorithm list:
        - dqn
        - dueling_ddqn
        - ppo
    '''
    algorithm_version = 'dueling_ddqn'

    # --- Setting up the game environment
    env = game.Game(ml_ai=True, render=False)
    env.game_initialize(ep_number=0)

    # --- Get the current state of the game by calling get_observation_X
    # FORMAT: city health, dx unit 1, dy unit 1, hp_norm unit 1, dx unit 2, dy unit 2, hp_norm unit 2, ...
    # with three units this will be a list of length 10
    state = np.array(env.get_observation())

    # --- Data list for plot
    attacker_r = np.array([])
    defender_r = np.array([])
    episodes = np.array([])

    # --- instantiate agents
    algorithm_dict = {
        'dqn': dqn.Agent,
        'dueling_ddqn': dueling_ddqn.Agent,
        'ppo': ppo.Agent
    }
    # Attacker action space
    attacker_agent = algorithm_dict[algorithm_version](state, 7)
    # Defender action space
    defender_agent  = algorithm_dict[algorithm_version](state, 7)

    # --- load saved model
    if enable_load:
        with open(f"./plots/{algorithm_version}.csv") as f:
            lines = list(csv.reader(f))
        lines = np.array(lines, float)
        attacker_r, defender_r = lines[:2]
        # Cumulative episodes is the number of episode from previous load. If there's nothing to load, start at 0
        cumulative_episodes = len(attacker_r)
        attacker_agent.load_model(f'attacker_{algorithm_version}')
        defender_agent.load_model(f'defender_{algorithm_version}')
        print("Continuing from last save data")
       
    else:
        cumulative_episodes = 0 

    training_start_time = time.time()
    episode_start = cumulative_episodes+1
    episode_end = cumulative_episodes+N_EPISODES+1
    for epoch in range(episode_start, episode_end):

        # --- Initialize the game by putting units and city on the playing field, etc.
        env.game_initialize(ep_number=epoch)
        state = np.array(env.get_observation())

        # --- Get start time
        s = time.time()
        for step in range(N_TURNS):
            done = False
            attacker_end_turn = False
            defender_end_turn = False
            attacker_reward = 0
            defender_reward = 0
            # print(f"Attacker turn")
            while True:
                if done or attacker_end_turn:
                    break
                 # --- Determine what action to take. 
                attacker_action = attacker_agent.act(state)
                # --- Perform that action in the environment
                # print(f"Attacker action: {attacker_action}, turn: {step}")
                next_state, attacker_reward, attacker_end_turn, done = env.step('attacker', attacker_action, attacker_reward)
                # --- Store state and action into memory
                attacker_agent.remember(state, next_state, attacker_action, attacker_reward, done)
                # --- Update the current state of the game
                state = np.array(next_state)
                
            # print(f"Defender turn")
            while True:
                if done or defender_end_turn:
                    break
                 # --- Determine what action to take. 
                defender_action = defender_agent.act(state)
                # --- Perform that action in the environment
                # print(f"Defender action: {defender_action}, turn: {step}")
                next_state, defender_reward, defender_end_turn, done = env.step('defender', defender_action, defender_reward)
                # --- Store state and action into memory
                defender_agent.remember(state, next_state, defender_action, defender_reward, done)
                # --- Update the current state of the game
                state = np.array(next_state)

            if done:
                break
        
        # --- Replay the agent past experience
        attacker_agent.replay()
        defender_agent.replay()
        # --- Store latest reward
        attacker_r = np.append(attacker_r, attacker_reward)
        defender_r = np.append(defender_r, defender_reward) 
        # --- Get end time
        e = time.time()
        print(f"Episode: {epoch}, time spent: {round(e-s, 2)}s")
        # --- Save model and data value every 1000 episodes or at the last episode
        if enable_save and (epoch % 100 == 0 or epoch == episode_end-1):
            attacker_agent.save_model(f'attacker_{algorithm_version}')
            defender_agent.save_model(f'defender_{algorithm_version}')

            with open(f"./plots/{algorithm_version}.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attacker_r)
                writer.writerow(defender_r)
        
    training_end_time = time.time()
    print(f"Training finished. Total elapsed time: {round(training_end_time-training_start_time, 2)}s")

    # --- Rewards vs episode plot
    def moving_average(x, w=10):
        avg = np.convolve(x, np.ones(w), 'valid') / w
        avg = np.append(avg, np.repeat(np.nan, w-1))
        return avg

    plt.plot(moving_average(attacker_r), label='Attacker rewards')
    plt.plot(moving_average(defender_r), label='Defender rewards')
    plt.title(f"{algorithm_version} Rewards vs Episodes")
    plt.legend() 
    if enable_save:
        plt.savefig(f"plots/rewards_vs_episodes_{algorithm_version}.png")

    plt.show()
    

            





