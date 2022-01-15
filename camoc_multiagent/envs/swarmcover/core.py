import numpy as np

import pettingzoo.mpe._mpe_utils as mpe_utils

class Target(mpe_utils.Landmark):
    def __init__(self):
        super().__init__()
        self.size = 1.0 # targets are fat, we want to cover them
