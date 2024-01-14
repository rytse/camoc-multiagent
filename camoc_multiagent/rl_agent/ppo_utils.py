from datetime import datetime

import yaml

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


def train_ppo(
    env_train, env_eval, name, total_timesteps, verbose=0, preload_name=None, **kwargs
):
    if preload_name is None:
        preload_name = name
    env_eval.reset()

    # savebest_cb = EvalCallback(
    #     eval_env=env_eval,
    #     best_model_save_path=f"./policies/{name}_ppo_policy_best",
    #     log_path=f"./logs/{name}_eval_log.txt",
    #     eval_freq=10_000,
    #     deterministic=True,
    #     render=False,
    # )

    try:
        model = PPO.load(f"./policies/{preload_name}")
        model.set_env(env_train)
    except Exception:
        model = PPO(MlpPolicy, env_train, verbose=verbose, **kwargs)

    #model.learn(total_timesteps=total_timesteps, callback=savebest_cb)
    model.learn(total_timesteps=total_timesteps)
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


def read_hyperparams(name):
    hyperparams_dict = None
    with open(f"rl_agent/hyperparams/{name}_hyp.json", "r") as f:
        hyperparams_dict = yaml.safe_load(f)

    ret_dict = {
        "batch_size": hyperparams_dict["batch_size"],
        "clip_range": hyperparams_dict["clip_range"],
        "ent_coef": hyperparams_dict["ent_coef"],
        "gae_lambda": hyperparams_dict["gae_lambda"],
        "gamma": hyperparams_dict["gamma"],
        "learning_rate": hyperparams_dict["learning_rate"],
        "max_grad_norm": hyperparams_dict["max_grad_norm"],
        "n_epochs": hyperparams_dict["n_epochs"],
        "n_steps": hyperparams_dict["n_steps"],
        "sde_sample_freq": hyperparams_dict["sde_sample_freq"],
        "vf_coef": hyperparams_dict["vf_coef"],
        "policy_kwargs": {
            "log_std_init": hyperparams_dict["log_std_init"],
            "ortho_init": True,
        },
    }
    return ret_dict
