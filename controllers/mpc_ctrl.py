"""
Script attempt to try mpc control
Joshua Lee (joshhlee@stanford.edu)
10/23/2023
"""

import numpy as np 
import random

class MPC_Controller:
    def __init__(self, state_observer, goal):
        self.observer = state_observer
        self._u = None
        self.goal_state = goal
        self.command_history = np.zeros([20,8])

    def compute_dpos(self, state, goal):
        delta_pos = np.linalg.norm(goal[0:2] - state[0:2])
        delta_angle = goal[2] - state[2]

        return [delta_pos, delta_angle]

    def eval_input(self):
        for i in range(10000):
            command_seq = np.zeros(20)
            for j in range(20):
                sample = random.randint(0,80) # samples in base 3
                command_seq[j] = sample_2binary(sample) # returns a list 20 lines long of random sequence of thrusters

            # compute the cost for each command
            # estimate using the dynamics and current state how far from goal we will be
            # look at how long the thruster is on for, how much it switches
            # minimize that cost. 
        


    def eval_input(self):
        # we need to get the believed state and the actual state then we can control based on that

        self._u = np.array([[1,0,0,0,0,0,0,0]]).T
        return self._u

    def get_input(self):
        return self._u

def sample_2binary(num):
    base3_str = np.base_repr(num, base=3)
    if len(base3_str) < 4:
        base3_str = '0' * (4 - len(base3_str)) + base3_str
    lst = [int(d) for d in str(base3_str)]
    binary_number = ''
    for i in lst:
        leng = np.base_repr(i, base=2)
        if i !=2:
            leng = '0' * (2 - len(leng)) + leng
        binary_number += leng
    return binary_number
