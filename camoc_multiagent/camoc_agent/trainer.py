import sys

import numpy as np
import jax.numpy as jnp

from stable_baselines3 import PPO

from camoc_agent.camoc_rotatorcoverage import CAMOC_RotatorCoverage_Agent
from camoc_agent.manifold_utils import *
from envs.rotator_coverage import rotator_coverage_v0


# Load the environment (to pull data from!)
env = rotator_coverage_v0.env_eval()

world = env.env.env.env.world
NUM_AGENTS = world.n_agents
NUM_TARGETS = world.n_entities - NUM_AGENTS
MAX_SPEED = world.maxspeed
DT = world.dt

print(f"Num agents: {NUM_AGENTS}")
print(f"Num targets: {NUM_TARGETS}")

# Create CAMOC agent
cagent = CAMOC_RotatorCoverage_Agent(NUM_TARGETS, NUM_AGENTS, MAX_SPEED, DT)

# Load the pretrained RL agent
model = PPO.load("./policies/rotator_coverage_v0_2022_01_26_23_36")

# Sample a batch of trajectories
# for tidx in range(100):
# for tidx in range(20):
for tidx in range(200):
    
    if tidx % 10 == 0:
        print("Sampling trajectory {}".format(tidx))

    env.reset()
    for i, agent in enumerate(env.agent_iter()):
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        #print(f"Step: {i}")
        if not done:  # TODO slice off framestack sanely
            cagent.cagent.add_samples(jnp.asarray(np.array([obs[-20:]])), jnp.asarray(np.array([act])))
            #stuff = jnp.asarray(np.array([obs[-20:]])), jnp.asarray(np.array([act]))
        else:
            break

# Eval the CAMOC agent
# env.reset()
# num_zero_actions = 0
# for agent in env.agent_iter():
#     obs, reward, done, info = env.last()
# 
#     if done:
#         break
# 
#     act = cagent.cagent.policy(jnp.asarray(obs[-20:]))
# 
#     if not act.any():
#         num_zero_actions += 1
#         if num_zero_actions == 10:
#             break
# 
#     env.render()
#     env.step(act)
