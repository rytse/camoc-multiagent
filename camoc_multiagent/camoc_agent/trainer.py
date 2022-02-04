import sys
import pickle

import numpy as np
import jax.numpy as jnp

from stable_baselines3 import PPO

import camoc_agent.camoc_rotatorcoverage
from camoc_agent.manifold_utils import *
from envs.rotator_coverage import rotator_coverage_v0


# Load the environment (to pull data from!)
env = rotator_coverage_v0.env_eval()

world = env.env.env.env.world
NUM_AGENTS = world.n_agents
NUM_TARGETS = world.n_entities - NUM_AGENTS
MAX_SPEED = world.maxspeed
DT = world.dt

# Create CAMOC agent
cagent = camoc_agent.camoc_rotatorcoverage.CAMOC_RotatorCoverage_Agent(
    NUM_TARGETS, NUM_AGENTS, MAX_SPEED, DT
)

# Load the pretrained RL agent
model = PPO.load("./policies/rotator_coverage_v0_2022_01_26_23_36")

# Sample a batch of trajectories
for tidx in range(200):
    if tidx % 10 == 0:
        print("Sampling trajectory {}".format(tidx))

    env.reset()

    obs_array = jnp.empty((NUM_AGENTS, cagent.obs_size))
    act_array = jnp.empty((NUM_AGENTS, cagent.act_size))

    for i, agent in enumerate(env.agent_iter()):
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)

        if not done:  # TODO slice off framestack sanely
            obs_array.at[i, :].set(obs[-20:])
            act_array.at[i, :].set(act)
        else:
            break

    cagent.add_samples(obs_array, act_array)

# Save agent
# with open("camoc_agent/policies/camoc_agent.pickle", "wb") as fi:
#     pickle.dump(cagent, fi)

# Eval the CAMOC agent
cagent.aggregate_samples()
env.reset()
num_zero_actions = 0
for agent in env.agent_iter():
    obs, reward, done, info = env.last()

    if done:
        break

    act = cagent.policy(jnp.asarray(obs[-20:]))

    env.render()
    env.step(act.ravel())
