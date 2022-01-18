from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from fast_simple_spread import FastSimpleEnv, FastScenario
import supersuit as ss
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
import numpy as np
from pettingzoo.mpe.scenarios.simple_spread import Scenario as SimpleSpreadScenario
from pettingzoo.mpe._mpe_utils.core import Agent, World, Landmark




class raw_env(FastSimpleEnv):
    def __init__(self, N=5, local_ratio=0.25, max_cycles=25, continuous_actions=True):
        assert 0. <= local_ratio <= 1., "local_ratio is a proportion. Must be between 0 and 1."
        scenario = FastScenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata['name'] = "swarm_cover_v1"


def preprocess_train(env):

    def _preprocess_train(**kwargs):
        nenv = env(**kwargs)
        nenv = ss.pettingzoo_env_to_vec_env_v1(nenv)
        nenv = ss.concat_vec_envs_v1(nenv, 1, num_cpus=1, base_class='stable_baselines3')

        return nenv

    return _preprocess_train


def preprocess_eval(env):

    def _preprocess_eval(**kwargs):
        nenv = env(**kwargs)
        return nenv

    return _preprocess_eval


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

env = preprocess_train(parallel_env)
#eval_env = preprocess_eval(env)

model = PPO(MlpPolicy, env, verbose=3, gamma=0.95, n_steps=256,
            ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202,
            max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3,
            batch_size=256)
