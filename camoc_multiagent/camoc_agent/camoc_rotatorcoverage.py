from functools import partial

#from jax import jit
import numpy as np
import jax.numpy as jnp
#from jax.lax import dynamic_slice, dynamic_update_slice

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
        prealloc_size: int =  50_000_000,
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

        return self.__class__._obs2mfd_cm(
            obs,
            self.obs_size,
            num_obs,  # we pass entire obs array, even the empty part
            self.mfd_size,
            self.m_d2t,
            self.m_a2t,
            self.m_d2a,
            self.m_s2a,
            self.o_d2t,
            self.o_a2t,
            self.o_d2a,
            self.o_s2a,
            self.max_speed,
        )

    @staticmethod
    #@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    def _obs2mfd_cm(
        obs,
        obs_len,
        num_obs,
        mfd_size,
        m_d2t,
        m_a2t,
        m_d2a,
        m_s2a,
        o_d2t,
        o_a2t,
        o_d2a,
        o_s2a,
        max_speed,
    ):
        """
        Convert an observation to its manifold coordinates. This is a jax jited
        and uses XLA slicing.

        Args:
          obs: observation
          mfd_buf: buffer to store the manifold coordinates
          obs_len: length of the observation
          m_d2t: index of the distance to targets slice in the manifold coords
          m_a2t: index of the angle to targets slice in the manifold coords
          m_d2a: index of the distance to agents slice in the manifold coords
          m_s2a: index of the speed to agents slice in the manifold coords
          o_d2t: index of the distance to targets slice in the observation coords
          o_a2t: index of the angle to targets slice in the observation coords
          o_d2a: index of the distance to agents slice in the observation coords
          o_s2a: index of the speed to agents slice in the observation coords
          max_speed: maximum speed of agents
        Returns:
            manifold coordinates
        """

        # Distance to target is not changed
        # obs.shape = (num samples, num obs coords)
        # d2t_mfd = dynamic_slice(obs, (0, o_d2t), (num_obs, o_a2t - o_d2t))
        # ^ wrong indexing!
        d2t_mfd = obs[:num_obs, o_d2t:o_a2t]

        # Angle to target is converted into alternating cos(theta), sin(theta)
        # obs_a2t = dynamic_slice(obs, (0, o_a2t), (num_obs, o_d2a - o_a2t)).flatten()
        # ^ wrong indexing!
        obs_a2t = obs[:num_obs, o_a2t:o_d2a].flatten()
        a2t_mfd = np.empty((num_obs, 2))
        a2t_mfd[:, 0] = np.cos(obs_a2t)
        a2t_mfd[:, 1] = np.sin(obs_a2t)

        # Distance to agents is not changed
        # d2a_mfd = dynamic_slice(obs, (0, o_d2a), (num_obs, o_s2a - o_d2a))
        # ^ wrong indexing!
        d2a_mfd = obs[:num_obs, o_d2a:o_s2a]

        # Speed is slacked with the halfinterval2slack function
        # obs_s2a = dynamic_slice(obs, (0, o_s2a), (num_obs, obs_len - o_s2a))
        # ^ wrong indexing!
        obs_s2a = obs[:num_obs, o_s2a:]
        s2a_mfd = halfinterval2slack(obs_s2a, max_speed)

        # Perform buffer updates
        mfd_buf = np.empty((num_obs, mfd_size))  # pre-allocate output buffer
        #        mfd_buf = dynamic_update_slice(mfd_buf, d2t_mfd, (0, m_d2t))
        #        mfd_buf = dynamic_update_slice(mfd_buf, a2t_mfd, (0, m_a2t))
        #        mfd_buf = dynamic_update_slice(mfd_buf, d2a_mfd, (0, m_d2a))
        #        mfd_buf = dynamic_update_slice(mfd_buf, s2a_mfd, (0, m_s2a))
        mfd_buf[:, m_d2t:m_a2t] = d2t_mfd
        mfd_buf[:, m_a2t:m_d2a] = a2t_mfd
        mfd_buf[:, m_d2a:m_s2a] = d2a_mfd 
        mfd_buf[:, m_s2a:] = s2a_mfd

        return mfd_buf

    #@partial(jit, static_argnums=(0, 3))
    def act2tpm(self, act, obs, num_obs):
        return self.__class__._act2tpm_cm(
            act,
            obs,
            self.mfd_size,
            num_obs,
            self.o_d2t,
            self.o_a2t,
            self.m_d2t,
            self.m_a2t,
            self.dt,
        )

    @staticmethod
    #@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
    def _act2tpm_cm(act, obs, mfd_len, num_obs, o_d2t, o_a2t, m_d2t, m_a2t, dt):

        # This assumes only one target for now TODO make it work for multiple
        d2t = obs[:num_obs, o_d2t].ravel()
        a2t = obs[:num_obs, o_a2t].ravel()

        # x-y distance to target
        Dx = d2t * np.cos(a2t)
        Dy = d2t * np.sin(a2t)
        # change in x-y due to action
        dx = act[:num_obs, 0].ravel() * np.cos(act[:num_obs, 1].ravel())
        dy = act[:num_obs, 0].ravel() * np.sin(act[:num_obs, 1].ravel())
        Dd = np.sqrt(np.square(Dx - dx) + np.square(Dy - dy))  # clsoer/farther

        dtheta = a2t - act[:num_obs, 1].ravel()

        v = np.empty((num_obs, mfd_len))
        v[:, m_d2t] = Dd * dt
        v[:, m_a2t] = np.cos(dtheta)
        v[:, m_a2t + 1] = np.sin(dtheta)  # the rest are zeros

        return v

    #@partial(jit, static_argnums=(0,))
    def tpm2act(self, tpm, obs):
        """
        Convert a TPM to an action. Calls the jax jited static method _tpm2act_cm.

        Args:
          tpm: TPM
          obs: observation
        Returns:
            action
        """

        return self.__class__._tpm2act_cm(
            tpm, obs, self.o_d2t, self.o_a2t, self.m_d2t, self.m_a2t, self.dt
        )

    @staticmethod
    #@partial(jit, static_argnums=(2, 3, 4, 5, 6))
    def _tpm2act_cm(tpm, obs, o_d2t, o_a2t, m_d2t, m_a2t, dt):
        #breakpoint()
        d_old = obs[:, o_d2t]
        phi_old = obs[:, o_a2t]
        #d_new = d_old + tpm[:, m_d2t] * dt
        d_new = d_old + tpm[m_d2t] * dt
        #phi_new = phi_old + tpm[:, m_a2t] * dt
        phi_new = phi_old + tpm[m_a2t] * dt

        l_x_new = d_new * np.cos(phi_new)
        l_y_new = d_new * np.sin(phi_new)
        l_x_old = d_old * np.cos(phi_old)
        l_y_old = d_old * np.sin(phi_old)
        dlx = l_x_old - l_x_new
        dly = l_y_old - l_y_new

        theta_new = np.arctan2(dly, dlx)
        speed_new = np.sqrt(np.square(dlx) + np.square(dly)) / dt

        act = np.array([theta_new, speed_new])
        return act

    #@partial(jit, static_argnums=(0,))
    def g_constr(self, x):
        return self.__class__._g_constr_cm(
            x, self.m_a2t, self.m_d2a, self.m_s2a, self.max_speed
        )

    @staticmethod
    #@partial(jit, static_argnums=(1, 2, 3, 4))
    def _g_constr_cm(x, m_a2t, m_d2a, m_s2a, max_speed):
        a2t_constr = np.sum(np.square(x[:, m_a2t:m_d2a]), axis=1) - 1  # = 0

        xs = x[:, m_s2a::3]
        alphas = x[:, m_s2a + 1 :: 3]
        betas = x[:, m_s2a + 2 :: 3]

        alpha_constrs = np.exp(alphas) + xs + max_speed - 1  # = 0
        beta_constrs = np.exp(betas) - xs - 1  # = 0

        all_constrs = (
            np.square(a2t_constr)
            + np.sum(np.square(alpha_constrs), axis=1)
            + np.sum(np.square(beta_constrs), axis=1)
        )

        return all_constrs[0]
