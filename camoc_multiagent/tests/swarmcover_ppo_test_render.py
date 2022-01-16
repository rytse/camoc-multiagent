from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from envs.swarmcover import swarm_cover_v1
import supersuit as ss

# Rendering
env = swarm_cover_v1.env()
env = ss.frame_stack_v1(env, 3)
model = PPO.load("./policies/swarmcover_ppo_policy")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()
