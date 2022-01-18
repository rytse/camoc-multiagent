from collections import OrderedDict

from stable_baselines3 import PPO

from camoc_agent.camoc_agent import CAMOCAgent
from camoc_agent.manifold_utils import *
from camoc_agent.camoc_agent import CAMOCAgent
from envs.swarmcover import swarm_cover_v1

# Load the environment and pretrained RL agent
env = swarm_cover_v1.train_env()
model = PPO.load("./policies/swarmcover_ppo_policy")
model.set_env(env)

envschema = {
    "dists_to_targets": {
        "native_size": NUM_TARGETS,
        "mfd_size": NUM_TARGETS,
        "conversion_lambda": identity_factory(),
    },

    "angles_to_targets": {
        "native_size": NUM_TARGETS,
        "mfd_size": NUM_TARGETS * 2,
        "conversion_lambda": angle2cart_factory(),
    },

    "dist_to_agents": {
        "native_size": NUM_AGENTS - 1,
        "mfd_size": NUM_AGENTS - 1,
        "conversion_lambda": identity_factory(),
    },

    "speed_to_agents": {
        "native_size": NUM_AGENTS - 1,
        "mfd_size": (NUM_AGENTS - 1) * 3,
        "conversion_lambda": halfinterval2slack_factory(MAX_SPEED),
    },
}


def g_constr(obs_mfd):
    """
    Constraints on the observation manifold.

        g_constr(q) = 0 for all q in MFD
    """

    return 0


cagent = CAMOCAgent(env, model)
