import numpy as np

from pettingzoo.mpe._mpe_utils.core import Agent, World, Landmark
#from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe.scenarios.simple_spread import Scenario as SimpleSpreadScenario
#from core import Target

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

        # Generate mesh that approximates the target
        theta = np.linspace(0, 2*np.pi, N_THETA)
        xx = LANDMARK_SIZE * np.cos(theta)
        yy = LANDMARK_SIZE * np.sin(theta)
        self.target_mesh = np.rollaxis(np.array([xx, yy]), 1)

        return world


    def global_reward(self, world):
        rew = 0
        in_place_mesh = self.target_mesh - world.landmarks[0].state.p_pos 

        for a in world.agents:
            rew -= np.min(np.linalg.norm(in_place_mesh - a.state.p_pos))

        return rew
