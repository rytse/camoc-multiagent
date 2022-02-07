import PIL

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from envs.rotator_coverage import rotator_coverage_v0


env = rotator_coverage_v0.env_eval()
# model = PPO.load("./policies/rotator_coverage_v0_ppo_policy_best/best_model")
model = PPO.load("./policies/rotator_coverage_v0_f2_2022_02_07_03_45")

frame_list = []
env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()
    frame_list.append(PIL.Image.fromarray(env.render(mode="rgb_array")))

frame_list[0].save(
    "ppo_out.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0
)
