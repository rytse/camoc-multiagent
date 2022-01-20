from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from envs.swarmcover import swarm_cover_v1
import supersuit as ss

env = swarm_cover_v1.train_env()


model = PPO(
    MlpPolicy,
    env,
    verbose=3,
    n_steps=256,
    batch_size=256,
    n_epochs=5,
    gamma=0.9,
    ent_coef=0.0030735679138708203,
    learning_rate=0.032934272464911644,
    vf_coef=0.546885902480104,
    max_grad_norm=5,
    gae_lambda=0.92,
    clip_range=0.4,
    sde_sample_freq=-1,
    policy_kwargs={"log_std_init": -1.60025077202246, "ortho_init": True},
)

model.learn(total_timesteps=20000)

model.save("./policies/swarmcover_ppo_policy")

# Rendering
env = swarm_cover_v1.eval_env()
model = PPO.load("./policies/swarmcover_ppo_policy")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()
