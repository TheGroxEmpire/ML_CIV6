# run_game.py
# Basic file how to run the game and control it with an AI

import numpy as np
import game
import random
import dqn

if __name__ == '__main__':


    # --- Set up your algorithm here
    N_EPISODES = 100
    N_EPISODE_STEPS = 20

    # --- Setting up the game environment
    env = game.Game(ml_ai=True, render=True)
    env.game_initialize(ep_number=0)
    # --- Get the current state of the game by calling get_observation_X
    # FORMAT: city health, dx unit 1, dy unit 1, hp_norm unit 1, dx unit 2, dy unit 2, hp_norm unit 2, ...
    # with three units this will be a list of length 10
    state = env.get_observation()

    #instantiate agents
    attacker_agent = dqn.DQN(state)
    defender_agent  = dqn.DQN(state)

    for epoch in range(N_EPISODES):

        # --- Initialize the game by putting units and city on the playing field, etc.
        env.game_initialize(ep_number=epoch)
        state = env.get_observation()
        attacker_r = []
        defender_r = []
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

            # --- Store latest reward
            attacker_r.append(attacker_reward)
            defender_r.append(defender_reward)       

            # --- Break the step loop if the game is done, aka the city is dead
            if done:
                attacker_r = np.mean(attacker_r)
                defender_r = np.mean(defender_r)
                print("Episode number: ", epoch, ", attacker rewards average: ", attacker_r, ", defender rewards average: ", defender_r, "turns taken: ", step)

                # --- Save stats
                file = open("./plot/dqn-stats", 'a')
                file.write(f"Episode number: {N_EPISODES}, attacker rewards average: {attacker_r}, defender rewards average: {defender_r}")
                file.close()
                break

            





