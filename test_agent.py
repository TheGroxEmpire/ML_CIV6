from email import policy
import pettingzoo_env

import os
import numpy as np
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.dqn as dqn
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class MyCallbacks(DefaultCallbacks):

    def on_train_result(self, *, algorithm, result: dict, **kwargs):

        result["custom_metrics"]["policy_reward_mean"] = {

            "attacker": result["policy_reward_mean"].get("attacker", np.nan),

            "defender": result["policy_reward_mean"].get("defender", np.nan),

        }

if __name__ == '__main__':
    os.environ["TUNE_ORIG_WORKING_DIR"] = os.getcwd()
    
    algorithm_version = 'DQN'
    comment_suffix = "a(3w2s)-d(2w1s)_default"
    checkpoint_dir = "logs\a(3w2s)-d(2w1s)_default\DQN\DQN_my_env_4b191_00000_0_2022-09-08_13-05-52\checkpoint_033523\checkpoint-33523"

    config = DQNConfig()

    def env_creator(max_turn=20, render_mode="show"):
        env = pettingzoo_env.PettingZooEnv(max_turn, render_mode)
        return env

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    
    register_env("my_env",lambda config: PettingZooEnv(env_creator()))

    config.multi_agent(

        policies={pid: (None, obs_space, act_space, {}) for pid in

                  test_env.env.agents},

        policy_mapping_fn=(lambda agent_id, episode, **kwargs: agent_id),

        )  

    config.num_gpus = 0
    config.log_level = "INFO"
    config.rollouts(num_rollout_workers=3)
    config.environment(env="my_env")
    

    config = config.to_dict()

    agent = dqn.DQN(config=config, env="my_env")

    done = False
    obs = test_env.reset()
    episode_reward = 0
    agents = ["attacker", "defender"]
    while True:
        action = {}
        for agent_id, agent_obs in obs.items():
            policy_id = config['multiagent']['policy_mapping_fn'](agent_id, None)
            action[agent_id] = agent.compute_single_action(agent_obs, policy_id=policy_id)
            
        obs, reward, done, info = test_env.step(action[agent_id])

        