from datetime import datetime

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO


def train_ppo(env, name, total_timesteps, **kwargs):
    try:
        model = PPO.load(f"./policies/{name}")
        model.set_env(env)
    except Exception:
        model = PPO(MlpPolicy,
                    env,
                    verbose=3,
                    n_steps=256,
                    batch_size=256,
                    n_epochs=5,
                    **kwargs
                )

    model.learn(total_timesteps=total_timesteps)
    now = datetime.now()
    model.save(f"./policies/{name}_{now.strftime('%Y_%m_%d_%H_%M')}")

    return model


def eval_ppo(env, name):
    model = PPO.load("./policies/swarmcover_ppo_policy")

    env.reset()
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        env.render()
