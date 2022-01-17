import random
import gym
from gym.spaces import Box
import numpy as np
from typing import Optional
class ComplexSpreadEnv(gym.Env):
    """
    Observation space:
    X position, Y position, theta orientation
    (0, 100),   (0,100),    (0, 2pi)

    Action Space:
    left-right, up-down, rotate
    (-1, 1), (-1,1), (-np.pi/90, -np.pi/90)
    """
    def __init__(self):
        self.observation_space = Box(
            low=np.array([0,0,0]),
            high=np.array([100, 100, 2*np.pi])
        )
        self.action_space = Box(
            low = np.array([-1,-1,-np.pi/90]),
            high= np.array([1,1, np.pi/90])
        )
    
        self.viewer = None
        self.state = None

    def step(self, action):
        assert self.action_space.contains(action), "ComplexSpreadEnv step: passed action not in action space"
        # actions represent the motion to take
        # can be changed later to physics-y
        self.state = self.state + action

        # 



    def reset(self, seed: Optional[int] = None) -> np.array:
        super().reset(seed=seed)
        # copied from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        # our observation space is 4 tho, not 4
        self.state = self.np_random.uniform(low=-.05, high=.05, size=(3,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)