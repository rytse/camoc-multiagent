import numpy as np

from pettingzoo.mpe._mpe_utils.core import Agent, World, Landmark
from pettingzoo.mpe.scenarios.simple_spread import Scenario as SimpleSpreadScenario

class Scenario(SimpleSpreadScenario):
    def make_world(self, N=3, AGENT_SIZE=0.15, LANDMARK_SIZE=1.5, N_THETA=10):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = 1   # swarm cover only wants to cover one target
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = AGENT_SIZE
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.size = LANDMARK_SIZE 
            landmark.collide = True  # you can bounce off if you get too close
            landmark.movable = False

        return world


    def global_reward(self, world):
        dist_penalty = 0
        angles = np.zeros(len(world.agents))

        for i, a in enumerate(world.agents):
            diff_v = a.state.p_pos - world.landmarks[0].state.p_pos

            dist_penalty -= np.linalg.norm(diff_v)
            angles[i] = np.arctan2(diff_v[1], diff_v[0])

        return -dist_penalty / (np.var(angles) + 1e-6)
