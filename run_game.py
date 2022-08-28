import pettingzoo_env

import os
from copy import deepcopy

from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.contrib.maddpg.maddpg_policy import MADDPGTFPolicy
from ray.rllib.contrib.maddpg.maddpg import MADDPGTrainer
from ray.tune.logger import pretty_print
from ray import tune

if __name__ == '__main__':
    os.environ["TUNE_ORIG_WORKING_DIR"] = os.getcwd()
    
    algorithm_name = 'PPO'
    comment_suffix = "a(3w2s)-d(2w1s)_default"
    N_EPISODE = 1
    CHECKPOINT_FREQ = 0

    trainer_dict = {
        'DQN': PPOTrainer,
        'PPO': DQNTrainer,
        'MADDPG': MADDPGTrainer
    }

    policy_dict = {
        'DQN': DQNTFPolicy,
        'PPO': PPOTFPolicy,
        'MADDPG': MADDPGTFPolicy
    }        

    config = deepcopy(get_trainer_class(algorithm_name)._default_config)

    def env_creator(max_turn=20, render_mode="show"):
        return pettingzoo_env.PettingZooEnv(max_turn, render_mode)

    register_env("my_env", lambda config: PettingZooEnv(env_creator())) 

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space 

    policies = {
        "attacker": (
            policy_dict[algorithm_name],
            obs_space,
            act_space,
            {},
        ),
        "defender": (
            policy_dict[algorithm_name],
            obs_space,
            act_space,
            {},
        ),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return agent_id

    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["log_level"] = "INFO"
    config["num_workers"] = 1

    config_attacker = config
    config_defender = config

    config_attacker["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["attacker"],
    }

    config_defender["multiagent"] = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["defender"],
    }

    attacker_agent = trainer_dict[algorithm_name](env="my_env", config=config_attacker)
    defender_agent = trainer_dict[algorithm_name](env="my_env", config=config_defender)

    for epoch in range(N_EPISODE):
       result_attacker = attacker_agent.train()
       print(pretty_print(result_attacker))
       result_defender = defender_agent.train()
       print(pretty_print(result_defender))