from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from pettingzoo.mpe import simple_spread_v2
import supersuit as ss

env = simple_spread_v2.parallel_env()
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")

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

model.learn(total_timesteps=20000)

model.save("./policies/simple_spread_v2_ppo_policy")

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
