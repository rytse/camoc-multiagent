from abc import ABC, abstractmethod

import jax.numpy as np

# import numpy as np
from jax import grad, jit, vmap


class CAMOCAgent:
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
        obs2mfd,
        tpm2act,
        act2tpm,
        g_constr,
        g_grad_constr=None,
        n_iters=2,
        prealloc_size=50_000_000,
    ):
        """
        Create a new CAMOC agent.

        Args:
            obs2mfd: map from observations to observation space manifold
            tpm2act: map from tangent space -> actions
            act2tpm: map from actions -> tangent space
            g_constr: level-set constraint function g(x)
            g_grad_constr: g'(x) (if None, it will be jax.grad(g_constr))
            n_iters: number of Newton's method iterations to perform (default 2)
        """

        self.obs2mfd = obs2mfd
        self.tpm2act = tpm2act
        self.act2tpm = act2tpm

        self.g_constr = g_constr
        if g_grad_constr is None:
            self.g_grad_constr = grad(g_constr)
        else:
            self.g_grad_constr = g_grad_constr
        self.n_iters = n_iters

        self.prealloc_size = prealloc_size

        self._obs = np.empty((prealloc_size, obs_size))
        self._act = np.empty((prealloc_size, act_size))

        self._obs_idx = 0
        self._act_idx = 0

        self._mpoints = None
        self._tmvecs = None

    def add_samples(self, observations, actions):
        """
        Add new samples to the agent's memory.

        Args:
            observations: list of observations
            actions: list of actions
        """

        # Resize as needed
        if self._obs.shape[0] <= self._obs_idx + observations.size:
            old_obs = self._obs
            old_act = self._act

            self._obs = np.zeros(
                (old_obs.shape[0] + self.prealloc_size, old_obs.shape[1])
            )
            self._act = np.zeros(
                (old_act.shape[0] + self.prealloc_size, old_act.shape[1])
            )

            self._obs.at[: old_obs.shape[0], :].set(old_obs)
            self._act.at[: old_act.shape[0], :].set(old_act)

        self._obs.at[self._obs_idx : self._obs_idx + observations.size, :].set(
            observations
        )
        self._act.at[self._act_idx : self._act_idx + actions.size, :].set(actions)

        self._obs_idx += observations.size
        self._act_idx += actions.size

        self._mpoints = None
        self._tmvecs = None

    def aggregate_samples(self):
        self._mpoints = self.obs2mfd(self._obs[: self._obs_idx, :])
        self._tmvecs = self.act2tpm(self._act[: self._act_idx, :])

    def policy(self, obs):
        """
        Generate an action for a given observation.

        Args:
            obs: observation
        Returns:
            action
        """
        obs_m = self.obs2mfd(obs)
        idxs = self._find_nearest_simplex(obs_m)
        vhat = np.average(self._tmvecs[idxs], axis=0)
        vbar = self._project_onto_mfd(obs_m, vhat)
        v = self.tpm2act(vbar, obs_m)
        if np.isnan(v).any():
            breakpoint()
        return v

    def _find_nearest_simplex(self, mpoint):
        """
        Find the nearest simplex to a given point (nearest 3 points).

        Args:
            mpoint: point on the manifold
        Returns:
            idxs: indices of nearest simplex (nearest 3 points)
        """

        differences = self._mpoints - mpoint
        dists = np.linalg.norm(differences, axis=1)
        # order = np.argpartition(dists, 3)
        order = np.argsort(dists)
        return order[0:3]

    def _project_onto_mfd(self, x, vhat):
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
        for _ in range(self.n_iters):
            ggc = self.g_grad_constr(x)
            dld = -self.g_constr(vhat + ggc * ld) / np.inner(ggc, ggc)
            ld += dld

            if np.isnan(ggc).any() or np.isnan(dld):
                breakpoint()

        vbar = vhat + self.g_grad_constr(vhat) * ld

        if np.isnan(vbar).any():
            breakpoint()
        return vbar
