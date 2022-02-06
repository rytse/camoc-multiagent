from abc import ABC, abstractmethod
from functools import partial
import numpy as np

import jax.numpy as jnp
from jax import grad


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
        self.obs = np.empty((prealloc_size, obs_size))
        self.act = np.empty((prealloc_size, act_size))
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

        self.obs[self.obs_idx:self.obs_idx+obs.shape[0], :] = obs
        self.act[self.act_idx:self.act_idx+act.shape[0], :] = act


        self.obs_idx += obs.shape[0]
        self.act_idx += act.shape[0]


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
        obs_m = self.obs2mfd(np.array([obs]), 1)
        idxs = self.find_nearest_simplex(obs_m)
        vhat = np.average(self.tmvecs[idxs], axis=0)
        vbar = self.project_onto_mfd(obs_m, vhat)
        v = self.tpm2act(vbar, obs_m)

        if np.isnan(v).any():
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
        dists = np.linalg.norm(differences, axis=1)
        k = 3 # nearest 3 elements
        order = np.argpartition(dists, k)
        #breakpoint()
        #order = np.argsort(dists)
        return order[:k]

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
            x, vhat, self.g_constr, self._g_grad_constr, self.n_iters
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
        '''
        for _ in range(n_iters):
            dld = -g_constr(vhat + ggc * ld) / np.inner(ggc, ggc)
            ld += dld
        '''
        
        
        

        return vhat #+ g_grad_constr(np.array([vhat])) * ld

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

    @abstractmethod
    def _g_grad_constr(self, x):
        """
        I don't know ryan come up with some bullshit, big brain, spe
        """
        pass

