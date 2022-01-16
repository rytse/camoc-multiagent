import envs.swarmcover.swarm_cover_v1
from rl_agent.train_ppo import train_ppo

if __name__ == '__main__':
    train_ppo(
        envs.swarmcover.swarm_cover_v1.train_env,
        envs.swarmcover.swarm_cover_v1.eval_env,
        'swarm_cover_v1'
    )

