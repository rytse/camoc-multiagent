from datetime import datetime

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


def train_ppo(env_train, env_eval, name, total_timesteps, **kwargs):
    env_eval.reset()

    savebest_cb = EvalCallback(
        eval_env=env_eval,
        best_model_save_path=f"./policies/{name}_ppo_policy_best",
        log_path=f"./logs/{name}_eval_log.txt",
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    try:
        model = PPO.load(f"./policies/{name}")
        model.set_env(env_train)
    except Exception:
        model = PPO(MlpPolicy, env_train, verbose=0, **kwargs)

    model.learn(total_timesteps=total_timesteps, callback=savebest_cb)
    now = datetime.now()
    model.save(f"./policies/{name}_{now.strftime('%Y_%m_%d_%H_%M')}")

    return model


def eval_ppo(env_eval, name):
    model = PPO.load(f"./policies/{name}_ppo_policy")
    model = PPO.load(f"./policies/{name}_ppo_policy")

    env_eval.reset()
    for agent in env_eval.agent_iter():
        obs, reward, done, info = env_eval.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env_eval.step(act)
        env_eval.render()
