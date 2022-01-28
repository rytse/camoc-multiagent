from abc import ABC, abstractmethod

# import jax.numpy as np
import numpy as np
from jax import grad, jit, vmap


class CAMOCGManifold:
    """
    Manifold defined by a level curve of a function g(x). CAMOC agents
    represent their observation space as the product manifold of several of
    these child manifolds. The Newton's method projection step is well-defined
    on each of these child manifolds, but since there is no one global g(x)
    function for which the entire observation space is a level curve, we
    must project onto each child manifold individually.
    """

    def __init__(self, g_constr, g_grad_constr, h_inv, x_slice, v_slice, n_iters=2):
        """
        Create a new level-set manifold object.

        Args:
            g_constr: level-set constraint function g(x)
            g_grad_constr: g'(x) (if None, it will be jax.grad(g_constr))
            h_inv: map from action coords -> tangent vector coords
            x_slice: indices of parent manifold coordinates to use
            v_slice: indicies of parent manifold tangent vectors to use
            n_iters: number of Newton's method iterations to perform
        """

        self._g_constr = g_constr
        if g_grad_constr is None:
            self._g_grad_constr = grad(g_constr)
        else:
            self._g_grad_constr = g_grad_constr
        self._h_inv = h_inv
        self.x_slice = x_slice
        self.v_slice = v_slice
        self._n_iters = n_iters

    def project(self, x, vhat_g):
        """
        Take a vector vhat attached to the manifold at point x and project it
        onto T_x M, and return that vector vbar.

        Args:
            x: point on manifold
            vhat_g: action (parent tangent space coords)
        Returns:
            vbar: projected "action" (parent tangent space coords)
        """

        vhat = vhat_g[self.v_slice]  # child tangent vector coords

        ld = 0  # Lagrange multiplier lambda
        for _ in range(self._n_iters):
            ggc = self._g_grad_constr(x[self.x_slice])
            dld = -self._g_constr(vhat + ggc * ld) / np.inner(ggc, ggc)
            ld += dld

        vbar = vhat + self._g_grad_constr(vhat) * ld
        return vbar


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

    def __init__(self, g_mfds, obs2mfd, tpm2act, act2tpm):
        """
        Create a new CAMOC agent.

        Args:
            g_mfds: list of level-set manifolds
            obs2mfd: map from observations to observation space manifold
            tpm2act: map from tangent space -> actions
            act2tpm: map from actions -> tangent space
        """

        self.g_mfds = g_mfds

        self.obs2mfd = obs2mfd
        self.tpm2act = tpm2act
        self.act2tpm = act2tpm

        self._mpoints = np.array([])  # samples' locations on the manifold
        self._tmvecs = np.array([])  # samples' associated tangent vectors

    def add_samples(self, observations, actions):
        """
        Add new samples to the agent's memory.

        Args:
            observations: list of observations
            actions: list of actions
        """

        if self._mpoints.size == 0:
            self._mpoints = self.obs2mfd(observations)
            self._tmvecs = self.act2tpm(actions)
        else:
            self._mpoints = np.vstack((self._mpoints, self.obs2mdf(observations)))
            self._tmvecs = np.vstack((self._tmvecs, self.act2tpm(actions)))

    def policy(self, obs):
        """
        Generate an action for a given observation.

        Args:
            obs: observation
        Returns:
            action
        """
        obs = self.obs2mfd(obs)
        idxs = self._find_nearest_simplex(obs)
        vhat = np.average(self._tmvecs[idxs], axis=0)
        vbar = self._project_onto_mfd(obs, vhat)
        v = self.tpm2act(vbar)
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
        order = np.argpartition(dists, 3)
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

        vbar = np.zeros_like(vhat)
        for g_mfd in self.g_mfds:
            vbar[g_mfd.v_slice] = g_mfd.project(x, vhat)
        return vbar


class CAMOC_Env_Agent(ABC):
    """
    Abstract base class for a CAMOC environment-agent pair.
    """

    def __init__(self):
        self.agent = CAMOCAgent(
            self.make_g_mfds, self.obs2mfd, self.tpm2act, self.act2tpm
        )

    @abstractmethod
    def make_g_mfds(self):
        pass

    @abstractmethod
    def obs2mfd(self):
        pass

    @abstractmethod
    def tpm2act(self):
        pass

    @abstractmethod
    def act2tpm(self):
        pass
