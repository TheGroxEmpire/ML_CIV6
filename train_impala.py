import pettingzoo_env

import os
import numpy as np
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.registry import register_env
from ray.rllib.algorithms.impala import ImpalaConfig
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
    
    algorithm_version = 'IMPALA'
    comment_suffix = "a(3w2s)-d(2w1s)_default"

    config = ImpalaConfig()

    def env_creator(max_turn=20, render_mode="hide"):
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

    config.num_gpus = 1
    config.log_level = "INFO"
    config.environment(env="my_env")
    config.rollouts(num_rollout_workers=3)
    
    config = config.to_dict()
     
    register_env("env", lambda config: PettingZooEnv(env_creator()))
    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    result = tune.Tuner(algorithm_version,
                param_space=config,
                run_config=air.RunConfig(
                    stop={"timesteps_total": 100000000},
                    checkpoint_config=air.CheckpointConfig(

                        checkpoint_frequency=10000,
                        checkpoint_at_end=True,
                    ),

                    local_dir=f"models/{comment_suffix}",

                    progress_reporter=CLIReporter(

                    metric_columns={

                        "training_iteration": "training_iteration",

                        "time_total_s": "time_total_s",

                        "timesteps_total": "timesteps",

                        "episodes_this_iter": "episodes_trained",

                        "custom_metrics/policy_reward_mean/attacker": "m_reward_a",

                        "custom_metrics/policy_reward_mean/defender": "m_reward_d",

                        "episode_reward_mean": "mean_reward_sum",
                    },
                    sort_by_metric=True,
                    ),
                ),
            ).fit()
