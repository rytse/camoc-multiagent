import sys
import pickle

import numpy as np
import PIL

from stable_baselines3 import PPO

import camoc_agent.camoc_rotatorcoverage
from camoc_agent.manifold_utils import *
from envs.rotator_coverage import rotator_coverage_v0
import time
from eli_env import MultiRotatorEnvironment
import torch

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

# Load the pretrained RL agent
# model = PPO.load("./policies/rotator_coverage_v0_2022_01_26_23_36")
model = PPO.load("./policies/rotator_coverage_v0_f2_2022_02_07_03_45")

# Sample a batch of trajectories
s = time.time()

n_envs = 5_000
env = MultiRotatorEnvironment(NUM_AGENTS, 1, n_envs)

# breakpoint()

# act = model.predict(vobs.clone().cpu())[0]
# cagent.add_samples(vobs.clone().cpu(), act)

obs = env.reset()
vobs = obs.view(obs.shape[0] * obs.shape[1], obs.shape[2])
for i in range(50):
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
    # if obs.max() > 30:
    #    breakpoint()
    # env.render()
    # obs = env.world.observation()
    vobs = obs.view(obs.shape[0] * obs.shape[1], obs.shape[2])


# Eval the CAMOC agent
print("Aggregating")
cagent.aggregate_samples()
env.reset()
num_zero_actions = 0


def test():

    frame_list = []

    env = rotator_coverage_v0.env_eval()
    env.reset()

    print("Starting cagent")
    for i, agent in enumerate(env.agent_iter()):
        print(f"Step: {i}")
        obs, reward, done, info = env.last()

        if done:
            break

        # act = cagent.policy(np.array(obs[-20:])).ravel()
        act = cagent.policy(obs).ravel()
        # breakpoint()
        # act = model.predict(obs, deterministic=True)[0]
        env.step(act)
        env.render()
        frame_list.append(PIL.Image.fromarray(env.render(mode="rgb_array")))

    frame_list[0].save(
        "camoc_out.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0
    )
    env.close()


test()


breakpoint()
