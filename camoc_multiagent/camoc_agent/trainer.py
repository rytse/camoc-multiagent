import sys
import pickle

import numpy as np
import PIL

from stable_baselines3 import PPO

import torch

import camoc_agent.camoc_rotatorcoverage
from camoc_agent.manifold_utils import *
from envs.rotator_coverage import multi_rotator_coverage_v0
import time

# Load the environment (to pull data from!)
n_envs = 5_0
env = multi_rotator_coverage_v0.env_eval(10, 1, n_envs)

world = env.world
NUM_AGENTS = env.n_agents
NUM_TARGETS = env.n_landmarks
MAX_SPEED = world.max_speed
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

# Sample from PPO
tmp = """
obs = env.reset()
vobs = obs.view(obs.shape[0] * obs.shape[1], obs.shape[2])
for i in range(5):
    if torch.isnan(obs).any():
        breakpoint()
    cpv = vobs.cpu()
    act = model.predict(cpv, deterministic=True)[0]
    if np.isnan(act).any():
        breakpoint()
    cagent.add_samples(cpv, act)
    obs, _, _ = env.step(
        torch.tensor(act, device=torch.device("cuda")).view(n_envs, NUM_AGENTS, 2)
    )
    vobs = obs.view(obs.shape[0] * obs.shape[1], obs.shape[2])
    """


# Eval the CAMOC agent
print("Aggregating")
cagent.aggregate_samples()
env.reset()
num_zero_actions = 0

def test(i):

    frame_list = []

    env.reset()
    obs = env.reset()
    vobs = obs.view(obs.shape[0] * obs.shape[1], obs.shape[2])
    for i in range(1000):
        if torch.isnan(obs).any():
            breakpoint()

        cpv = vobs.cpu()
        act = cagent.policy(cpv.numpy())
        if np.isnan(act).any():
            breakpoint()
        obs, _, _ = env.step(
            torch.tensor(act, device=torch.device("cuda")).view(n_envs, NUM_AGENTS, 2)
        )

        env.render()
        frame_list.append(PIL.Image.fromarray(env.render(mode="rgb_array")))
        vobs = obs.view(obs.shape[0] * obs.shape[1], obs.shape[2])

    frame_list[0].save(
        f"camoc_out_{i}.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0
    )


for i in range(10):
    test(i)
