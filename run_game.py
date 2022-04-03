# run_game.py
# Basic file how to run the game and control it with an AI


import game
import random

if __name__ == '__main__':


    # --- Set up your algorithm here
    N_GAMES = 4
    N_GAME_STEPS = 20

    # --- Setting up the game environment
    env = game.Game(ml_ai=True, render=True)


    for epoch in range(N_GAMES):

        # --- Initialize the game by putting units and city on the playing field, etc.
        env.game_initialize(ep_number=epoch)

        # --- Get the current state of the game by calling get_observation_X
        # FORMAT: city health, dx unit 1, dy unit 1, hp_norm unit 1, dx unit 2, dy unit 2, hp_norm unit 2, ...
        # with three units this will be a list of length 10
        state = env.get_observation()


        for step in range(N_GAME_STEPS):

            # --- Determine what action to take. 
            # For 5 units it is one of 16807 possibilities (7^5)
            attacker_action = random.randint(0, 16807)
            # For 3 units it is one of 343 possibilities (7^3)
            defender_action = random.randint(0, 343)

            # --- Perform that action in the environment
            next_state, attacker_reward, defender_reward, done = env.step(attacker_action, defender_action)

            # --- Train your algo here



            # --- Break the step loop if the game is done, aka the city is dead
            if done:
                break




            # --- Update the current state of the game
            state = next_state





