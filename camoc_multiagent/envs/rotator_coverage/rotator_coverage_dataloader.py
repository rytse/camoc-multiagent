import sys
from torch.utils.data import Dataset


class RotatorCoverageDataloader(Dataset):
    def __init__(self, model, n_agents, n_targets, max_iter=500):
        self.model = model
        self.env = rotator_coverage_v0.env_eval()
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.max_iter = max_iter

    def __len__(self):
        sys.maxsize

    def __getitem__(self, index):
        obs_dim = self.n_targets * 4 + (self.n_agents - 1) * 2
        act_dim = 2

        X = torch.zeros(self.n_agents * obs_dim, self.max_iter)
        Y = torch.zeros(self.n_agents * (obs_dim + act_dim), self.max_iter)

        env.reset()
        for agent in env.agent_iter(self.max_iter):
            obs, reward, done, info = env.last()
            act = model.predict(obs, deterministic=True)[0] if not done else None
            env.step(act)
