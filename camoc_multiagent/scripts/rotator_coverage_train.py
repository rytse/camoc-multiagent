from envs.rotator_coverage import rotator_coverage_v0
from rl_agent.ppo_utils import train_ppo


env_train = rotator_coverage_v0.env_train()
env_eval = rotator_coverage_v0.env_train()

model = train_ppo(
    env_train,
    env_eval,
    "guess0_rotator_coverage_train",
    total_timesteps=10_000_000,
    n_steps=512,
    n_epochs=10,
    batch_size=32,
    gamma=0.995,
    ent_coef=0.000217028956545287,
    learning_rate=0.00398364348178975,
    vf_coef=0.812838013528646,
    max_grad_norm=5,
    gae_lambda=0.99,
    clip_range=0.1,
    sde_sample_freq=256,
    policy_kwargs={
        "log_std_init": -3.67097310146946,
        "ortho_init": True,
    },
)

env = rotator_coverage_v0.env_eval()
env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()
