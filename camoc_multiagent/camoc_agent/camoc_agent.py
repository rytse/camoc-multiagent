from abc import ABC, abstractmethod
from functools import partial
import jax.numpy as jnp

from jax import grad, jit, vmap
from jax.lax import dynamic_update_slice


class CAMOCAgent(ABC):
    """
    Agent that computes policies by interpolating and projecting actions
    sampled from another (pre-trained) agent.

    Agents have two outward-facing functions:
     1. Incorporating new samples
     2. Generating actions

    The observation space manifold is represented as the product of several
    level-set manifolds that can be projected onto via the Newton's method
    trick.

    The agent (this class) has notions of action coordinates and tangent vector
    coordinates. The child manifolds only have notions of the tangent vector
    coordinates, as those are the internal representation.

    Note: Currently agents memorize actions in their tangent vector
    coordinates. This representation obviously consumes more memory, since h
    is not one-to-one. However, it is faster at runtime. If memory becomes an
    issue, we can switch this.
    """

    def __init__(
        self,
        obs_size,
        act_size,
        mfd_size,
        n_iters=2,
        prealloc_size=50_000_000,
    ):
        """
        Create a new CAMOC agent.

        Args:
            obs_size: dimensionality of the observation space
            act_size: dimensionality of the action space
            mfd_size: dimensionality of the manifold
            n_iters: number of Newton's method iterations to perform (default 2)
            prealloc_size: number of samples to preallocate (default 50M)
        """

        self.obs_size = obs_size
        self.act_size = act_size
        self.mfd_size = mfd_size
        self.n_iters = n_iters
        self.prealloc_size = prealloc_size

        # Buffer of observations and actions in native coords
        self.obs = jnp.empty((prealloc_size, obs_size))
        self.act = jnp.empty((prealloc_size, act_size))
        self.obs_idx = 0
        self.act_idx = 0

        # Buffer of observations and actions in manifold coords. None until
        # aggregate_samples is called so that you don't accidentally use it
        # until you processed all added samples.
        self.mpoints = None
        self.tmvecs = None

        # Calculate the gradient of the constraint function only once
        self.g_grad_constr = grad(self.g_constr)

    def add_samples(self, obs, act):
        """
        Add new samples to the agent's memory.

        Args:
            obs: list of observations
            act: list of actions
        """

        # Resize as needed (not in place, this makes a copy!)
        #         if self.obs.shape[0] <= self.obs_idx + obs.shape[0]:
        #             old_obs = self.obs
        #             old_act = self.act
        #
        #             self.obs = jnp.zeros(
        #                 (old_obs.shape[0] + self.prealloc_size, old_obs.shape[1])
        #             )
        #             self.act = jnp.zeros(
        #                 (old_act.shape[0] + self.prealloc_size, old_act.shape[1])
        #             )
        #
        #             self.obs = self.obs.at[: old_obs.shape[0], :].set(old_obs)
        #             self.act = self.act.at[: old_act.shape[0], :].set(old_act)
        #
        #         #self.obs = _in_place_dus(self.obs, obs, (self.obs_idx, 0))
        #         #self.act = _in_place_dus(self.act, act, (self.act_idx, 0))

        self.obs, self.act = self.__class__._add_samples_cm(
            obs, act, self.obs, self.act, self.obs_idx
        )

        self.obs_idx += obs.shape[1]
        self.act_idx += act.shape[1]

    @staticmethod
    @jit
    def _add_samples_cm(obs, act, obs_buf, act_buf, buf_idx):
        obs_buf = dynamic_update_slice(obs_buf, obs, (buf_idx, 0))
        act_buf = dynamic_update_slice(act_buf, act, (buf_idx, 0))

        return obs_buf, act_buf

    def aggregate_samples(self):
        self.mpoints = self.obs2mfd(self.obs, self.obs_idx)
        self.tmvecs = self.act2tpm(self.act, self.obs, self.act_idx)

    def policy(self, obs):
        """
        Generate an action for a given observation.

        Args:
            obs: observation
        Returns:
            action
        """
        obs_m = self.obs2mfd(jnp.array([obs]), 1)
        idxs = self.find_nearest_simplex(obs_m)
        vhat = jnp.average(self.tmvecs[idxs], axis=0)
        vbar = self.project_onto_mfd(obs_m, vhat)
        v = self.tpm2act(vbar, obs_m)

        if jnp.isnan(v).any():
            breakpoint()

        return v

    def find_nearest_simplex(self, mpoint):
        """
        Find the nearest simplex to a given point (nearest 3 points).

        Args:
            mpoint: point on the manifold
        Returns:
            idxs: indices of nearest simplex (nearest 3 points)
        """

        differences = self.mpoints - mpoint
        dists = jnp.linalg.norm(differences, axis=1)
        order = jnp.argsort(dists)

        return order[0:3]

    def project_onto_mfd(self, x, vhat):
        """
        Project a vector vhat onto the manifold at point x by aggregating
        the projection of each child manifold.

        Args:
            x: point on the manifold
            vhat: vector to project
        Returns:
            vbar: projected vector
        """

        return self.__class__._project_onto_mfd_cm(
            x, vhat, self.g_constr, self.g_grad_constr, self.n_iters
        )

    @staticmethod
    def _project_onto_mfd_cm(x, vhat, g_constr, g_grad_constr, n_iters):
        """
        Project a vector vhat onto the manifold at point x by aggregating
        the projection of each child manifold.

        Args:
            x: point on the manifold
            vhat: vector to project
        Returns:
            vbar: projected vector
        """

        ld = 0  # Lagrange multiplier lambda
        ggc = g_grad_constr(x)

        for _ in range(n_iters):
            dld = -g_constr(vhat + ggc * ld) / jnp.inner(ggc, ggc)
            ld += dld

        return vhat + g_grad_constr(jnp.array([vhat])) * ld

    @abstractmethod
    def obs2mfd(self, obs):
        """
        Map an observation to its manifold coordinates

        Args:
            obs: observation
        Returns:
            mpoint: point on the manifold
        """
        pass

    @abstractmethod
    def act2tpm(self, act, obs):
        """
        Map an action to its tangent vector coordinates.

        Args:
            act: action
            obs: observation
        Returns:
            v: tangent vector
        """
        pass

    @abstractmethod
    def tpm2act(self, v, x):
        """
        Map a tangent vector in manifold coordinates to an action.

        Args:
            v: tangent vector
            x: point on the manifold
        Returns:
            action
        """
        pass

    @abstractmethod
    def g_constr(self, x):
        """
        Level-set constraint function g(x)

        Args:
            x: point on the manifold
        Returns:
            g(x): level-set constraint
        """
        pass
