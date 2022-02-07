from envs.rotator_coverage.rotator_coverage import (
    RotatorCoverageEnv,
    RotatorCoverageScenario,
)
import supersuit as ss
from pettingzoo.mpe._mpe_utils.simple_env import make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(RotatorCoverageEnv):
    def __init__(self, N=10, local_ratio=0.5, max_cycles=100, continuous_actions=True):
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = RotatorCoverageScenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata["name"] = "swarm_cover_v1"
        self.continuous_actions = True


def preprocess_train(env):
    def _preprocess_train(**kwargs):
        nenv = env(**kwargs)
        # nenv = ss.frame_stack_v1(nenv, 10)
        # nenv = ss.frame_stack_v1(nenv, 2)
        nenv = ss.pettingzoo_env_to_vec_env_v1(nenv)
        nenv = ss.concat_vec_envs_v1(
            nenv, 1, num_cpus=1, base_class="stable_baselines3"
        )

        return nenv

    return _preprocess_train


def preprocess_eval(env):
    def _preprocess_eval(**kwargs):
        nenv = env(**kwargs)
        # nenv = ss.frame_stack_v1(nenv, 10)
        # nenv = ss.frame_stack_v1(nenv, 2)
        return nenv

    return _preprocess_eval


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

env_train = preprocess_train(parallel_env)
env_eval = preprocess_eval(env)
