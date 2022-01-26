from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from envs.rotator_coverage import rotator_coverage_v0

env = rotator_coverage_v0.env_train()
model = PPO(
    MlpPolicy,
    env,
    verbose=3,
    gamma=0.95,
    n_steps=256,
    ent_coef=0.0905168,
    learning_rate=0.00062211,
    vf_coef=0.042202,
    max_grad_norm=0.9,
    gae_lambda=0.99,
    n_epochs=5,
    clip_range=0.3,
    batch_size=256,
)

model.learn(total_timesteps=100_000)
model.save("./policies/rotator_coverage_ppo_policy")

env = rotator_coverage_v0.env_eval()
env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()