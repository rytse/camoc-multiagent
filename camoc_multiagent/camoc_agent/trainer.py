from collections import OrderedDict

import numpy as np
from jax import grad, jit, vmap
import jax.numpy as jnp

from stable_baselines3 import PPO

from camoc_agent.camoc_agent import CAMOCAgent
from camoc_agent.manifold_utils import *
from camoc_agent.camoc_agent import CAMOCAgent
#from envs.swarmcover import swarm_cover_v1
import envs.custom_env.PREPROC as fast_simple_env_v0


# Load the environment (to pull data from!)
env = fast_simple_env_v0.env_eval()

world = env.env.env.world
NUM_AGENTS = world.n_agents
NUM_TARGETS = world.n_entities - NUM_AGENTS
MAX_SPEED = world.maxspeed * 2

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
    # Skip fields TODO remove this for final version for performance benefits
    c = envschema["dists_to_targets"]["mfd_size"]

    # Load angle data for constraint
    angle_to_targets = obs_mfd[c: c + envschema["angles_to_targets"]["mfd_size"]]
    #angle_to_targets = angle_to_targets.reshape([2, int(angle_to_targets.shape[0] / 2)])
    breakpoint()
    angle_to_targets = jnp.reshape(angle_to_targets, [2, int(angle_to_targets.shape[0] / 2)])


    # Skip fields
    c = 0
    c += envschema["angles_to_targets"]["mfd_size"]
    c += envschema["dist_to_agents"]["mfd_size"]

    # Load speed data for constraint
    speed_to_agents = obs_mfd[c: c + envschema["speed_to_agents"]["mfd_size"]]
    #speed_to_agents = speed_to_agents.reshape([3, int(speed_to_agents.shape[0] / 3)])
    speed_to_agents = jnp.reshape(speed_to_agents, [3, int(speed_to_agents.shape[0] / 3)])

    # Constraints
    angle_constr = np.linalg.norm(angle_to_targets, axis=0) - 1  # = 0
    speed_constr_alpha = np.power(speed_to_agents[1, :], 2) - speed_to_agents[0, :]
    speed_constr_beta = np.power(speed_to_agents[2, :], 2) - speed_to_agents[0, :]

    # Return vector of constrains. Usually g : M -> R, but since we have the
    # product manifold of many level-set manifolds, the projection onto the
    # product manifold is the same as the projection onto each individual one.
    all_constrs = np.zeros(angle_constr.shape[0] + speed_constr_alpha.shape[0] + speed_constr_beta.shape[0])
    all_constrs[: angle_constr.shape[0]] = angle_constr
    all_constrs[angle_constr.shape[0]: angle_constr.shape[0] + speed_constr_alpha.shape[0]] = speed_constr_alpha
    all_constrs[angle_constr.shape[0] + speed_constr_alpha.shape[0]:] = speed_constr_beta

    return all_constrs


# Initialize the CAMOC agent
cagent = CAMOCAgent(g_constr)

# Load the pretrained RL agent
model = PPO.load("./policies/cenv_ppo_policy")

# Sample a batch of trajectories
#for tidx in range(100):
for tidx in range(20):

    if tidx % 10 == 0:
        print("Sampling trajectory {}".format(tidx))

    env.reset()
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        # env.render()
        if not done:
            obs_mfd = obs_native2mfd(obs, envschema)
            cagent.add_samples(obs_mfd, act)
        else:
            break

# Eval the CAMOC agent
env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = cagent.policy(obs_native2mfd(obs, envschema))

    env.render()
    env.step(act)

