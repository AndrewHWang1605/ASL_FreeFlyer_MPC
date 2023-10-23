"""
Returns noisy "state estimate" from dynamics true state
"""

import numpy as np 

class StateObserver():
    def __init__(self, dynamics, mean = 0, sd = 0):
        """
        Init function for state observer

        Args:
            dynamics (Dynamics): Dynamics object instance
            mean (float, optional): Mean for gaussian noise. Defaults to 0. 
            sd (float, optional): standard deviation for gaussian noise. Defaults to 0.
        """
        self.dynamics = dynamics
        self.stateDimn = dynamics.stateDimn
        self.mean = mean
        self.sd = sd
        
    def get_state(self):
        """
        Returns a potentially noisy observation of the system state
        """
        print("state",self.dynamics.get_state())
        return self.dynamics.get_state() + np.random.normal(self.mean, self.sd, (self.stateDimn, 1))
    

