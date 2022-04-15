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
    N_EPISODES = 100000
    N_EPISODE_STEPS = 30
    '''Algorithm list:
        - Vanilla-DQN
        - Dueling-DQN
    '''
    algorithm_version = 'Dueling-DQN'

    # --- Setting up the game environment
    env = game.Game(ml_ai=True, render=False)
    env.game_initialize(ep_number=0)

    # --- Get the current state of the game by calling get_observation_X
    # FORMAT: city health, dx unit 1, dy unit 1, hp_norm unit 1, dx unit 2, dy unit 2, hp_norm unit 2, ...
    # with three units this will be a list of length 10
    state = env.get_observation()
    
    # --- Data list for plot. These get turned to numpy array
    attacker_r = []
    defender_r = []
    episodes = []

    # --- instantiate agents
    algorithm_dict = {
        'Vanilla-DQN': dqn.Vanilla_DQN,
        'Dueling-DQN': dqn.Dueling_DQN
    }
    # For attacker (5 units) it is one of 16807 possibilities (7^5)
    attacker_agent = algorithm_dict[algorithm_version](state, 16807)
    # For defender (3 units) it is one of 343 possibilities (7^3)
    defender_agent  = algorithm_dict[algorithm_version](state, 343)   

    # --- load saved model
    if enable_load:
        try:
            with open(f"./plots/{algorithm_version}.csv") as f:
                lines = list(csv.reader(f))
            lines = np.array(lines, float)
            attacker_r, defender_r = lines[:2]
            episodes = len(attacker_r)
            # Cumulative episodes is the number of episode from previous load. If there's nothing to load, start at 0
            cumulative_episodes = episodes
            attacker_agent.load_model(f'attacker_{algorithm_version}')
            defender_agent.load_model(f'defender_{algorithm_version}')
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
            attacker_action = attacker_agent.act(state)
            defender_action = defender_agent.act(state)

            # --- Perform that action in the environment
            #print(f"Attacker action: {attacker_action}, turn: {step}")
            #print(f"Defender action: {defender_action}, turn: {step}")
            next_state, attacker_reward, defender_reward, done = env.step(attacker_action, defender_action)
            
            # --- Store state and action into memory
            attacker_agent.remember(state, next_state, attacker_action, attacker_reward, done)
            defender_agent.remember(state, next_state, defender_action, defender_reward, done)
            
            # --- Update the current state of the game
            state = next_state      
            # --- Break the step loop if the game is done, aka the city is dead
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
        # --- Save model and data value every 10 episodes or at the last episode
        if enable_save and (epoch % 10 == 0 or epoch == episode_end-1):
            attacker_agent.save_model(f'attacker_{algorithm_version}')
            defender_agent.save_model(f'defender_{algorithm_version}')

            with open(f"./plots/{algorithm_version}.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attacker_r)
                writer.writerow(defender_r)
        
    training_end_time = time.time()
    print(f"Training finished. Total elapsed time: {round(training_end_time-training_start_time, 2)}s")

    # --- Rewards vs episode plot
    episodes = np.arange(0, episodes+1)
    plt.plot(episodes, attacker_r, label='Attacker rewards')
    plt.plot(episodes, defender_r, label='Defender rewards')
    plt.title('Rewards vs Episodes')
    plt.legend() 
    if enable_save:
        plt.savefig(f"plots/rewards_vs_episodes_{algorithm_version}.png")

    plt.show()

    

            





