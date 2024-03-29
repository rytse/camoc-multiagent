from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from pettingzoo.mpe import simple_spread_v2
import supersuit as ss

# Rendering
env = simple_spread_v2.env()
env = ss.frame_stack_v1(env, 3)
model = PPO.load("./policies/simple_spread_v2_ppo_policy")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()
