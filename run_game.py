import pettingzoo_env

import os
from copy import deepcopy

from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.agents.registry import get_trainer_class
from ray import tune

if __name__ == '__main__':
    os.environ["TUNE_ORIG_WORKING_DIR"] = os.getcwd()
    
    algorithm_version = 'DQN'
    comment_suffix = "a(3w2s)-d(2w1s)_default"

    config = deepcopy(get_trainer_class(algorithm_version)._default_config)

    def env_creator(max_turn=20, render_mode="hide"):
        env = pettingzoo_env.PettingZooEnv(max_turn=max_turn, render_mode=render_mode)
        return env

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    
    register_env("my_env",lambda config: PettingZooEnv(env_creator()))

    config["multiagent"] = {
        "policies": {
            "attacker": (None, obs_space, act_space, {}),
            "defender": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: agent_id,
    }   

    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["log_level"] = "INFO"
    config["num_workers"] = 2
    config["env"] = "my_env"
     
    register_env("env", lambda config: PettingZooEnv(env_creator()))
    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    tune.run(algorithm_version,
             name=algorithm_version,
             checkpoint_freq=1000,
             config=config)