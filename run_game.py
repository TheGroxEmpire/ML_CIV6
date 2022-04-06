# run_game.py
# Basic file how to run the game and control it with an AI

import csv
import time
import numpy as np
import game
import matplotlib.pyplot as plt
import dqn

if __name__ == '__main__':

    # --- Load / save setting
    enable_load = True
    enable_save = True
    
    # --- Set up your algorithm here
    N_EPISODES = 4996
    N_EPISODE_STEPS = 50
    algorithm_version = '2_Vanilla-DQN'

    # --- Setting up the game environment
    env = game.Game(ml_ai=True, render=True)
    env.game_initialize(ep_number=0)
    # --- Get the current state of the game by calling get_observation_X
    # FORMAT: city health, dx unit 1, dy unit 1, hp_norm unit 1, dx unit 2, dy unit 2, hp_norm unit 2, ...
    # with three units this will be a list of length 10
    state = env.get_observation()

    # --- instantiate agents
    attacker_agent = dqn.Vanilla_DQN(state, 16807)
    defender_agent  = dqn.Vanilla_DQN(state, 343)
    #attacker_agent = dqn.Dueling_DQN(state, 16807)
    #defender_agent  = dqn.Dueling_DQN(state, 343)

    # --- Data list for plot. These get turned to numpy array down
    attacker_r = []
    defender_r = []
    episodes = []

    # --- load saved model
    if enable_load:
        attacker_agent.load_model('attacker_v2_Vanilla-DQN')
        defender_agent.load_model('defender_v2_Vanilla-DQN')
        try:
            with open(f"./plots/v{algorithm_version}.csv") as f:
                lines = list(csv.reader(f))
            lines = np.array(lines, float)
            attacker_r, defender_r, episodes = lines[:3]
            # Cumulative episodes is number of episode from previous load. If there's nothing to load, start at 0
            cumulative_episodes = int(episodes[-1])
            print("Continuing from last save data")
        except:
            print("No save data to load")
            cumulative_episodes = 0
    else:
        cumulative_episodes = 0     

    training_start_time = time.time()
    episode_start = cumulative_episodes+1
    episode_end = cumulative_episodes+N_EPISODES+1
    for epoch in range(episode_start, episode_end):

        # --- Initialize the game by putting units and city on the playing field, etc.
        env.game_initialize(ep_number=epoch)
        state = env.get_observation()

        # --- Get start time
        s = time.time()
        for step in range(N_EPISODE_STEPS):

            # --- Determine what action to take. 
            # For 5 units it is one of 16807 possibilities (7^5)
            attacker_action = attacker_agent.act(state)
            # For 3 units it is one of 343 possibilities (7^3)
            defender_action = defender_agent.act(state)

            # --- Perform that action in the environment
            #print(f"Attacker action: {attacker_action}, turn: {step}")
            #print(f"Defender action: {defender_action}, turn: {step}")
            next_state, attacker_reward, defender_reward, done_attacker, done_defender = env.step(attacker_action, defender_action)
            
            # --- Store state and action into memory
            attacker_agent.remember(state, next_state, attacker_action, attacker_reward, done_attacker)
            defender_agent.remember(state, next_state, defender_action, defender_reward, done_defender)
            
            # --- Update the current state of the game
            state = next_state      
            # --- Break the step loop if the game is done, aka the city is dead
            if done_attacker or done_defender:
                break

        # --- Experience replay
        attacker_agent.replay()
        defender_agent.replay()
        # --- Store latest reward
        attacker_r = np.append(attacker_r, attacker_reward)
        defender_r = np.append(defender_r, defender_reward) 
        # --- Get end time
        e = time.time()
        print(f"Episode: {epoch}, time spent: {round(e-s, 2)}s")
        episodes = np.append(episodes, epoch)
        # --- Save model and rewards / episode value every 10 episodes or at the last episode
        if enable_save and epoch % 10 == 0 or epoch == episode_end-1:
            attacker_agent.save_model(f'attacker_v{algorithm_version}')
            defender_agent.save_model(f'defender_v{algorithm_version}')

            with open(f"./plots/v{algorithm_version}.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attacker_r)
                writer.writerow(defender_r)
                writer.writerow(episodes)
        
    training_end_time = time.time()
    print(f"Training finished. Total elapsed time: {round(training_end_time-training_start_time, 2)}s")
        
    plt.plot(episodes, attacker_r, label='Attacker rewards')
    plt.plot(episodes, defender_r, label='Defender rewards')
    plt.title('Rewards vs Episodes')
    plt.legend() 
    plt.show()
    if enable_save:
        plt.savefig(f"plots/rewards_vs_episodes_v{algorithm_version}.png")

    

            





