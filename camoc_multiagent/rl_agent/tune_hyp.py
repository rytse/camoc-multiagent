from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import optuna

from rl_agent.ppo_utils import train_ppo


# Linear scheduling for variable learning rate
def _linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


# Hold the optuna study with the environment and name passed in
def study(env_train, env_eval, name, train_timesteps, optimizer_timesteps):

    # Optuna objective function that has local access to env and name
    def objective(trial):
        batch_size = trial.suggest_categorical(
            "batch_size", [8, 16, 32, 64, 128, 256, 512]
        )
        n_steps = trial.suggest_categorical(
            "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        )
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])

        gamma = trial.suggest_categorical(
            "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
        )
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
        lr_schedule = "constant"

        # Learning rate schedule
        lr_schedule = "linear"  # always linear!

        ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
        # ent_coef = 0.0905168

        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
        gae_lambda = trial.suggest_categorical(
            "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
        )
        max_grad_norm = trial.suggest_categorical(
            "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
        )
        vf_coef = trial.suggest_uniform("vf_coef", 0, 1)

        # gSDE (continuous actions)
        log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
        sde_sample_freq = trial.suggest_categorical(
            "sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256]
        )
        # Orthogonal initialization
        ortho_init = True  # MA-PPO says to always orthogonally initialize!

        if lr_schedule == "linear":
            learning_rate = _linear_schedule(learning_rate)

        # Set up the kwargs
        ppo_hyperparams = {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "gae_lambda": gae_lambda,
            "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
            "sde_sample_freq": sde_sample_freq,
            "policy_kwargs": dict(
                log_std_init=log_std_init,
                ortho_init=ortho_init,
            ),
        }

        model = train_ppo(env_train, env_eval, name, train_timesteps, **ppo_hyperparams)

        return model._logger.name_to_value["train/loss"]

    study_name = "hyptune-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, load_if_exists=True
    )
    study.optimize(objective, n_trials=optimizer_timesteps)

    return study.best_params
