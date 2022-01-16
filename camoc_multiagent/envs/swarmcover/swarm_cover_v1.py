from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env

import supersuit as ss

from .scenarios.swarm_cover import Scenario


class raw_env(SimpleEnv):
    def __init__(self, N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False):
        assert 0. <= local_ratio <= 1., "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata['name'] = "swarm_cover_v1"


def preprocess_train(env):

    def _preprocess_train(**kwargs):
        nenv = env(**kwargs)
        nenv = ss.frame_stack_v1(nenv, 3)
        nenv = ss.pettingzoo_env_to_vec_env_v1(nenv)
        nenv = ss.concat_vec_envs_v1(nenv, 8, num_cpus=8, base_class='stable_baselines3')

        return nenv

    return _preprocess_train


def preprocess_eval(env):

    def _preprocess_eval(**kwargs):
        nenv = env(**kwargs)
        nenv = ss.frame_stack_v1(nenv, 3)
        return nenv

    return _preprocess_eval


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

train_env = preprocess_train(parallel_env)
eval_env = preprocess_eval(env)
