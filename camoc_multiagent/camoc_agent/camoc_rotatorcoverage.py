from camoc_agent.camoc_agent import CAMOCAgent
from camoc_agent.manifold_utils import *


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
            self._obs2mfd, self._tpm2act, self._act2tpm, self._g_constr
        )

    def add_samples(self, observations, actions):
        self.cagent.add_samples(observations, actions)

    def policy(self, obs):
        return self.cagent.policy(obs)

    def _obs2mfd(self, obs):
        x = np.zeros(self.mfd_len)

        x[self.m_d2t : self.m_a2t] = obs[self.o_d2t : self.o_a2t]
        x[self.m_a2t : self.m_d2a : 2] = np.cos(obs[self.o_a2t : self.o_d2a])
        x[self.m_a2t + 1 : self.m_d2a : 2] = np.sin(obs[self.o_a2t : self.o_d2a])
        x[self.m_d2a : self.m_s2a] = obs[self.o_d2a : self.o_s2a]
        x[self.m_s2a :] = halfinterval2slack(obs[self.o_s2a :], self.max_speed)

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
        v[0] = Dd * self.dt  # this one gets integrated

        dtheta = a2t - act[1]
        v[1] = np.cos(dtheta)
        v[2] = np.sin(dtheta)

        return v  # other components have zero change in expected value

    def _g_constr(self, v):
        a2t_constr = np.linalg.norm(v[self.m_a2t : self.m_d2a]) - 1  # = 0

        x, alpha, beta = v[self.m_s2a :]
        s2a_alpha_constr = alpha * alpha + x - self.max_speed  # = 0
        s2a_beta_constr = beta * beta - x  # = 0

        constr_v = np.array([a2t_constr, s2a_alpha_constr, s2a_beta_constr])

        return np.sum(np.power(constr_v))  # only 0 when all components 0
