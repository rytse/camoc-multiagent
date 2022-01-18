#import jax.numpy as np
import numpy as np
from jax import grad, jit, vmap

class CAMOCAgent:
    '''
    Agent that computes policies by interpolating and projecting actions
    sampled from another (pre-trained) agent.

    Agents have two outward-facing functions:
     1. Incorporating new samples
     2. Generating actions
    '''

    def __init__(self, g_constr, g_grad_constr=None, newton_iters=2,
            _project_onto_actions=None):
        self._g_constr = g_constr
        if g_grad_constr is None:
            self._g_grad_constr = grad(g_constr)
        else:
            self._g_grad_constr = g_grad_constr

        self._newton_iters = newton_iters
        self._project_onto_actions = _project_onto_actions

        self._mpoints = np.array([])  # samples' locations on the manifold
        self._tmvecs = np.array([])  # samples' associated tangent vectors


    def add_samples(self, observations, actions):
        if self._mpoints.size == 0:
            self._mpoints = observations
            self._tmvecs = actions
        else:
            self._mpoints = np.vstack((self._mpoints, observations))
            self._tmvecs = np.vstack((self._tmvecs, actions))


    def policy(self, obs):
        idxs = self._find_nearest_simplex(obs)
        vhat = np.average(self._tmvecs[idxs], axis=0)
        vbar = self._project_onto_mfd(vhat)
        
        if not self._project_onto_actions is None:
            return self._project_onto_actions(vbar)
        
        return vbar


    def _find_nearest_simplex(self, mpoint):
        differences = self._mpoints - mpoint
        dists = np.linalg.norm(differences, axis=1)
        order = np.argpartition(dists, 3)

        return order[0:3]


    def _project_onto_mfd(self, vhat):

        breakpoint()

        ld = 0
        for i in range(self._newton_iters):
            ggc = self._g_grad_constr(vhat) 
            dld = -self.g_constr(vhat + ggc * ld) / np.inner(ggc, ggc)
            ld += dld

        return vhat + self.g_grad_constr(vhat) * ld

