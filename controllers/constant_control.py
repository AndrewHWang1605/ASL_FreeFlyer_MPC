"""
Script to supply constant control signal (for debugging)
Andrew Wang (aw1605@stanford.edu)
10/20/2023
"""

import numpy as np 

class ConstantController:
    def __init__(self, state_observer):
        self.observer = state_observer
        self._u = None

    def eval_input(self):
        self._u = np.array([[0,1,0,0,1,0,0,0]]).T
        return self._u

    def get_input(self):
        return self._u
