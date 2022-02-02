from camoc_agent.camoc_agent import CAMOCAgent
from camoc_agent.manifold_utils import *
import jax.numpy as np


class CAMOC_RotatorCoverage_Agent:
    """
    A CAMOC agent for the RotatorCoverage environment.

    Coordinate layouts:

      obs: [dists_to_targets, angles_to_targets, dist_to_agents, speed_to_agents]
      action: [theta, speed]

      x: [dist_to_targets <NUM_TARGETS>,
          angle2cart(angles_to_targets) <NUM_TARGETS*2>,
          dist_to_agents <NUM_AGENTS-1>,
          halfinterval2slack(speed_to_agents) <(NUM_AGENTS-1)*3>]
      v:
    """

    def __init__(self, num_targets, num_agents, max_speed, dt):
        self.num_targets = num_targets
        self.num_agents = num_agents
        self.max_speed = max_speed
        self.dt = dt

        # Pre-compute indices for the various slices of the observation coords
        self.obs_len = num_targets * 2 + (num_agents - 1) * 2
        self.o_d2t = 0
        self.o_a2t = self.o_d2t + num_targets
        self.o_d2a = self.o_a2t + num_targets
        self.o_s2a = self.o_d2a + num_agents - 1
        # Pre-compute indices for the various slices of the manifold coords
        self.mfd_len = (
            num_targets + num_targets * 2 + (num_agents - 1) + (num_agents - 1) * 3
        )
        self.m_d2t = 0
        self.m_a2t = self.m_d2t + num_targets
        self.m_d2a = self.m_a2t + num_targets * 2
        self.m_s2a = self.m_d2a + (num_agents - 1)

        # Create the CAMOC agent
        self.cagent = CAMOCAgent(
            self.obs_len,
            2,
            self.mfd_len,
            self._obs2mfd,
            self._tpm2act,
            self._act2tpm,
            self._g_constr,
        )

    def add_samples(self, observations, actions):
        self.cagent.add_samples(observations, actions)

    def aggregate_samples(self):
        self.cagent.aggregate_samples()

    def policy(self, obs):
        return self.cagent.policy(obs)

    def _obs2mfd(self, obs):
        x = np.zeros(self.mfd_len)

        x.at[self.m_d2t : self.m_a2t].set(obs[self.o_d2t : self.o_a2t])
        x.at[self.m_a2t : self.m_d2a : 2].set(np.cos(obs[self.o_a2t : self.o_d2a]))
        x.at[self.m_a2t + 1 : self.m_d2a : 2].set(np.sin(obs[self.o_a2t : self.o_d2a]))
        x.at[self.m_d2a : self.m_s2a].set(obs[self.o_d2a : self.o_s2a])
        x.at[self.m_s2a :].set(halfinterval2slack(obs[self.o_s2a :], self.max_speed))

        return x

    def _tpm2act(self, tpm, obs):
        d_old = obs[self.o_d2t]
        phi_old = obs[self.o_a2t]
        d_new = d_old + tpm[self.m_d2t] * self.dt
        phi_new = phi_old + tpm[self.m_a2t] * self.dt

        l_x_new = d_new * np.cos(phi_new)
        l_y_new = d_new * np.sin(phi_new)
        l_x_old = d_old * np.cos(phi_old)
        l_y_old = d_old * np.sin(phi_old)
        dlx = l_x_old - l_x_new
        dly = l_y_old - l_y_new

        theta_new = np.arctan2(dly, dlx)
        speed_new = np.linalg.norm(np.array([dlx, dly])) / self.dt

        return np.array([theta_new, speed_new])

    def _act2tpm(self, act, obs):
        v = np.zeros(self.mfd_len)

        # This assumes only one target for now TODO make it work for multiple
        d2t = obs[self.o_d2t]
        a2t = obs[self.o_a2t]

        Dx = d2t * np.cos(a2t)  # x-y distance to target
        Dy = d2t * np.sin(a2t)
        dx = act[0] * np.cos(act[1])  # change in x-y due to action
        dy = act[0] * np.sin(act[1])
        Dd = np.linalg.norm(np.array([Dx - dx, Dy - dy]))  # closer/farther

        dtheta = a2t - act[1]

        v.at[self.m_d2t].set(Dd * self.dt)
        v.at[self.m_a2t].set(np.cos(dtheta))
        v.at[self.m_a2t + 1].set(np.sin(dtheta))  # the rest are zeros

        return v

    def _g_constr(self, x):
        a2t_constr = np.sum(np.square(x[self.m_a2t : self.m_d2a])) - 1  # = 0

        xs = x[self.m_s2a :: 3]
        alphas = x[self.m_s2a + 1 :: 3]
        betas = x[self.m_s2a + 2 :: 3]

        # alpha_constrs = alphas * alphas + xs - self.max_speed  # = 0
        # beta_constrs = betas * betas - xs  # = 0
        alpha_constrs = np.exp(alphas) + xs + self.max_speed - 1  # = 0
        beta_constrs = np.exp(betas) - xs - 1  # = 0

        all_constrs = (
            np.square(a2t_constr)
            + np.sum(np.square(alpha_constrs))
            + np.sum(np.square(beta_constrs))
        )

        return all_constrs
