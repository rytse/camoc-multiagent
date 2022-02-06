import sys
import pickle

import numpy as np


from stable_baselines3 import PPO

import camoc_agent.camoc_rotatorcoverage
from camoc_agent.manifold_utils import *
from envs.rotator_coverage import rotator_coverage_v0
import time

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
s = time.time()
for tidx in range(1):
    if tidx % 10 == 0:
        print("Sampling trajectory {}".format(tidx))

    env.reset()

    for i, agent in enumerate(env.agent_iter()):
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)

        if not done:  # TODO slice off framestack sanely
            cagent.add_samples(np.array([obs[-20:]]), np.array([act]))
        else:
            break

# Eval the CAMOC agent
cagent.aggregate_samples()
env.reset()
num_zero_actions = 0
for agent in env.agent_iter():
    obs, reward, done, info = env.last()

    if done:
        break

    act = cagent.policy(np.array(obs[-20:]))
    env.render()
    env.step(act.ravel())
