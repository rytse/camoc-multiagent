from envs.rotator_coverage import rotator_coverage_v0
from rl_agent.ppo_utils import train_ppo, read_hyperparams

NAME = "rotator_coverage_v0"
NUM_TIMESTEPS = int(1e5)
EVAL_RENDER = True

print(f"NUM TIMESTEPS: {NUM_TIMESTEPS}")

env_train = rotator_coverage_v0.env_train()
env_eval = rotator_coverage_v0.env_train()

hyp = read_hyperparams(NAME)

model = train_ppo(
    env_train,
    env_eval,
    NAME,
    NUM_TIMESTEPS,
    verbose=3,
    preload_name=f"{NAME}_2022_01_26_23_36",
    **hyp,
)

if EVAL_RENDER:
    env = rotator_coverage_v0.env_eval()
    env.reset()
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        env.render()
