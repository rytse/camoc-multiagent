import sys
import pickle

import numpy as np
import PIL

from stable_baselines3 import PPO

import camoc_agent.camoc_rotatorcoverage
from camoc_agent.manifold_utils import *
from envs.rotator_coverage import rotator_coverage_v0
import time

# Load the environment (to pull data from!)
env = rotator_coverage_v0.env_eval()

world = env.env.env.world
NUM_AGENTS = world.n_agents
NUM_TARGETS = world.n_entities - NUM_AGENTS
MAX_SPEED = world.maxspeed
DT = world.dt

# Create CAMOC agent
cagent = camoc_agent.camoc_rotatorcoverage.CAMOC_RotatorCoverage_Agent(
    NUM_TARGETS, NUM_AGENTS, MAX_SPEED, DT
)

# Load previously saved model
try:
    obs_loaded = np.load("camoc_obs_traj.npy")
    act_loaded = np.load("camoc_act_traj.npy")
    cagent.add_samples(obs_loaded, act_loaded)
except FileNotFoundError:
    print("No saved model found. Training from scratch.")

# Load the pretrained RL agent
model = PPO.load("./policies/rotator_coverage_v0_f2_2022_02_07_03_45")

no_sample = """
# Sample a batch of trajectories
s = time.time()
for tidx in range(50):
    if tidx % 10 == 0:
        print("Sampling trajectory {}".format(tidx))

    env.reset()

    for i, agent in enumerate(env.agent_iter()):
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)

        if not done:  # TODO slice off framestack sanely
            cagent.add_samples(np.array([obs]), np.array([act]))
        else:
            break

# Save sample points
np.save("camoc_obs_traj.npy", cagent.obs[: cagent.obs_idx, :])
np.save("camoc_act_traj.npy", cagent.act[: cagent.act_idx, :])
"""

NUM_SPARSE = 20


def run_env(i):
    # Eval the CAMOC agent
    sample_idxs = np.random.permutation(np.arange(cagent.obs_idx))[:NUM_SPARSE]
    cagent.obs = cagent.obs[sample_idxs, :]
    cagent.act = cagent.act[sample_idxs, :]
    cagent.obs_idx = cagent.act_idx = NUM_SPARSE
    cagent.aggregate_samples()

    env.reset()
    num_zero_actions = 0
    frame_list = []
    counter = 0
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()

        if done:
            break

        act = cagent.policy(obs).ravel()

        env.step(act)
        env.render()
        frame_list.append(PIL.Image.fromarray(env.render(mode="rgb_array")))

    frame_list[0].save(
        f"camoc_out_{i}.gif",
        save_all=True,
        append_images=frame_list[1:],
        duration=3,
        loop=0,
    )


for i in range(10):
    run_env(i)
