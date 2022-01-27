from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from envs.rotator_coverage import rotator_coverage_v0


env = rotator_coverage_v0.env_eval()
model = PPO.load("./policies/rotator_coverage_v0_ppo_policy_best/best_model")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()
