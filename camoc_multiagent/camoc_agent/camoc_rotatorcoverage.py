from functools import partial

# from jax import jit
import numpy as np
import jax.numpy as jnp

# from jax.lax import dynamic_slice, dynamic_update_slice

from camoc_agent.camoc_agent import CAMOCAgent
from camoc_agent.manifold_utils import *


class CAMOC_RotatorCoverage_Agent(CAMOCAgent):
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

    def __init__(
        self,
        num_targets: int,
        num_agents: int,
        max_speed: float,
        dt: float,
        n_iters: int = 2,
        prealloc_size: int = 50_000_000,
    ):
        """
        Initialize a RotatorCoverage agent.

        Args:
          num_targets: number of targets
          num_agents: number of agents
          max_speed: maximum speed of agents
          dt: time step
          n_iters: number of iterations to run the dynamics
          prealloc_size: size of the preallocated buffers
        """

        super().__init__(
            num_targets * 2 + (num_agents - 1) * 2,
            2,
            num_targets + num_targets * 2 + (num_agents - 1) + (num_agents - 1) * 3,
            n_iters,
            prealloc_size,
        )

        # Save rotator_coverage specific parameters
        self.num_targets = num_targets
        self.num_agents = num_agents
        self.max_speed = max_speed
        self.dt = dt

        # Pre-compute indices for the various slices of the observation coords
        self.o_d2t = 0
        self.o_a2t = self.o_d2t + num_targets
        self.o_d2a = self.o_a2t + num_targets
        self.o_s2a = self.o_d2a + num_agents - 1

        # Pre-compute indices for the various slices of the manifold coords
        self.m_d2t = 0
        self.m_a2t = self.m_d2t + num_targets
        self.m_d2a = self.m_a2t + num_targets * 2
        self.m_s2a = self.m_d2a + (num_agents - 1)

    def obs2mfd(self, obs, num_obs):
        """
        Convert an observation to its manifold coordinates. Calls the jax jited
        static method _obs2mfd_cm.

        Args:
          obs: observation
        Returns:
          manifold coordinates
        """

        # Distance to target is not changed
        # obs.shape = (num samples, num obs coords)
        # d2t_mfd = dynamic_slice(obs, (0, o_d2t), (num_obs, o_a2t - o_d2t))
        # ^ wrong indexing!
        d2t_mfd = obs[:num_obs, self.o_d2t : self.o_a2t]

        # Angle to target is converted into alternating cos(theta), sin(theta)
        # obs_a2t = dynamic_slice(obs, (0, o_a2t), (num_obs, o_d2a - o_a2t)).flatten()
        # ^ wrong indexing!
        obs_a2t = obs[:num_obs, self.o_a2t : self.o_d2a].flatten()
        a2t_mfd = np.empty((num_obs, 2))
        a2t_mfd[:, 0] = np.cos(obs_a2t)
        a2t_mfd[:, 1] = np.sin(obs_a2t)

        # Distance to agents is not changed
        # d2a_mfd = dynamic_slice(obs, (0, o_d2a), (num_obs, o_s2a - o_d2a))
        # ^ wrong indexing!
        d2a_mfd = obs[:num_obs, self.o_d2a : self.o_s2a]

        # Speed is slacked with the halfinterval2slack function
        # obs_s2a = dynamic_slice(obs, (0, o_s2a), (num_obs, obs_len - o_s2a))
        # ^ wrong indexing!
        obs_s2a = obs[:num_obs, self.o_s2a :] / self.max_speed
        if self.o_s2a > 2 or self.o_s2a < 0:
            pass
            # breakpoint()
        s2a_mfd = halfinterval2slack(obs_s2a, 2)

        # Perform buffer updates
        mfd_buf = np.empty((num_obs, self.mfd_size))  # pre-allocate output buffer
        #        mfd_buf = dynamic_update_slice(mfd_buf, d2t_mfd, (0, m_d2t))
        #        mfd_buf = dynamic_update_slice(mfd_buf, a2t_mfd, (0, m_a2t))
        #        mfd_buf = dynamic_update_slice(mfd_buf, d2a_mfd, (0, m_d2a))
        #        mfd_buf = dynamic_update_slice(mfd_buf, s2a_mfd, (0, m_s2a))
        mfd_buf[:, self.m_d2t : self.m_a2t] = d2t_mfd
        mfd_buf[:, self.m_a2t : self.m_d2a] = a2t_mfd
        mfd_buf[:, self.m_d2a : self.m_s2a] = d2a_mfd
        mfd_buf[:, self.m_s2a :] = s2a_mfd

        return mfd_buf

    def act2tpm(self, act, obs, num_obs):
        """
        Convert an action to its tangent vector coordinates.
        """

        # Get old triangle from observation
        D_old = obs[:num_obs, self.o_d2t : self.o_a2t]  # dist to target
        phi_old = obs[:num_obs, self.o_a2t : self.o_d2a]  # angle to target
        x_old = D_old * np.cos(phi_old)
        y_old = D_old * np.sin(phi_old)

        # Get translation triangle from action
        theta = np.array([act[:num_obs, 0]]).T
        s = np.array([act[:num_obs, 1]]).T  # dist of translation
        l_x = s * np.cos(theta)
        l_y = s * np.sin(theta)

        # Get new triangle
        x_new = x_old - l_x  # new dist to target
        y_new = y_old - l_y
        phi_new = np.arctan2(y_new, x_new)
        D_new = np.sqrt(x_new ** 2 + y_new ** 2)

        # Get effect on tangent vector coordinates
        d_theta = phi_new - phi_old
        d_D = D_new - D_old

        # Stick it in a tangent vector with all the necessary zeros
        # for the other coordinates
        v = np.empty((num_obs, self.mfd_size))
        v[:, self.m_d2t] = d_D.T
        v[:, self.m_a2t] = np.cos(d_theta).T
        v[:, self.m_a2t + 1] = np.sin(d_theta).T  # the rest are zeros

        return v

    def tpm2act(self, tpm, obs):
        """
        Convert a TPM to an action. Calls the jax jited static method _tpm2act_cm.

        Args:
          tpm: TPM
          obs: observation
        Returns:
            action
        """

        # Get old triangle from observation
        d_old = obs[:, self.o_d2t]
        phi_old = obs[:, self.o_a2t]

        # Decode current heading angle from the tangent vector coordinates
        d_theta_x = tpm[:, self.m_a2t]
        d_theta_y = tpm[:, self.m_a2t + 1]
        d_theta = np.arctan2(d_theta_y, d_theta_x)

        # Decode current distance to target from the tangent vector coordinates
        d_D = tpm[:, self.m_d2t]

        # Get new triangle
        d_new = d_D + d_old
        phi_new = phi_old + d_theta

        # Get side lengths of old and new target triangles
        x_old = d_old * np.cos(phi_old)
        y_old = d_old * np.sin(phi_old)
        x_new = d_new * np.cos(phi_new)
        y_new = d_new * np.sin(phi_new)

        # Get translation triangle
        l_x = x_new - x_old
        l_y = y_new - y_old
        theta = np.arctan2(l_y, l_x)
        s = np.sqrt(l_x ** 2 + l_y ** 2)

        act = np.array([np.clip(theta, -np.pi, np.pi), np.clip(s, 0, 1)])
        return act

    def g_constr(self, x):
        a2t_constr = np.sum(np.square(x[:, self.m_a2t : self.m_d2a]), axis=1) - 1  # = 0

        xs = x[:, self.m_s2a :: 3]
        alphas = x[:, self.m_s2a + 1 :: 3]
        betas = x[:, self.m_s2a + 2 :: 3]

        alpha_constrs = np.exp(alphas) + xs + 2 - 1  # = 0
        beta_constrs = np.exp(betas) - xs - 1  # = 0

        all_constrs = (
            np.square(a2t_constr)
            + np.sum(np.square(alpha_constrs), axis=1)
            + np.sum(np.square(beta_constrs), axis=1)
        )

        return all_constrs[0]

    def _g_grad_constr(self, x):
        """
        Compute the gradient of the constraint function.
        """

        # Angle to target vars
        theta_qs = x[:, self.m_a2t : self.m_d2a]
        partial_thetas = 4 * theta_qs * (np.sum(np.square(theta_qs), axis=1) - 1)

        # Speed vars
        xs = x[:, self.m_s2a :: 3]
        alphas = x[:, self.m_s2a + 1 :: 3]
        betas = x[:, self.m_s2a + 2 :: 3]
        partial_xs = 2 * ((np.exp(alphas) + xs + 2 - 1) + (np.exp(betas) - xs - 1))
        partial_alphas = 2 * np.exp(alphas) * (np.exp(alphas) + xs + 2 - 1)
        partial_betas = 2 * np.exp(betas) * (np.exp(betas) - xs - 1)

        # Other unused vars (zero gradients)
        partial_d2t = np.zeros_like(x[:, self.m_d2t : self.m_a2t])
        partial_d2a = np.zeros_like(x[:, self.m_d2a : self.m_s2a])

        # Concatenate all partial derivatives into the gradient
        g_grad = np.concatenate(
            (
                partial_d2t,
                partial_thetas,
                partial_d2a,
                partial_xs,
                partial_alphas,
                partial_betas,
            ),
            axis=1,
        )
        return g_grad
