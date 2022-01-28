from camoc_agent.camoc_agent import CAMOC_Env_Agent
from camoc_agent.manifold_utils import *


class CAMOC_RotatorCoverage_Agent(CAMOC_Env_Agent):
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

    def make_g_mfds(self):
        pass

    def obs2mfd(self):
        pass

    def tpm2act(self):
        pass

    def act2tpm(self):
        pass
