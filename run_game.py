# run_game.py
# Basic file how to run the game and control it with an AI

import numpy as np
import game
import matplotlib.pyplot as plt
import dqn

if __name__ == '__main__':


    # --- Set up your algorithm here
    N_EPISODES = 100
    N_EPISODE_STEPS = 50

    # --- Setting up the game environment
    env = game.Game(ml_ai=True, render=True)
    env.game_initialize(ep_number=0)
    # --- Get the current state of the game by calling get_observation_X
    # FORMAT: city health, dx unit 1, dy unit 1, hp_norm unit 1, dx unit 2, dy unit 2, hp_norm unit 2, ...
    # with three units this will be a list of length 10
    state = env.get_observation()

    # --- instantiate agents
    attacker_agent = dqn.DQN(state, 16807)
    defender_agent  = dqn.DQN(state, 343)

    # --- Rewards array for plot
    attacker_r = []
    defender_r = []
    for epoch in range(N_EPISODES):

        # --- Initialize the game by putting units and city on the playing field, etc.
        env.game_initialize(ep_number=epoch)
        state = env.get_observation()
        for step in range(N_EPISODE_STEPS):

            # --- Determine what action to take. 
            # For 5 units it is one of 16807 possibilities (7^5)
            attacker_action = attacker_agent.act(state)
            # For 3 units it is one of 343 possibilities (7^3)
            defender_action = defender_agent.act(state)

            # --- Perform that action in the environment
            next_state, attacker_reward, defender_reward, done = env.step(attacker_action, defender_action)
            
            # --- Store state and action into memory
            attacker_agent.remember(state, next_state, attacker_action, attacker_reward, done)
            
            # --- Update the current state of the game
            state = next_state      

            # --- Break the step loop if the game is done, aka the city is dead
            if done:
                break
        
        # --- Store latest reward
        attacker_r.append(attacker_reward)
        defender_r.append(defender_reward) 
        
    Episodes = np.arange(0, N_EPISODES, 1)
    plt.plot(Episodes, attacker_r, label='Attacker rewards')
    plt.plot(Episodes, defender_r, label='Defender rewards')
    plt.title('Rewards vs Episodes')
    plt.legend()
    plt.show()

            





