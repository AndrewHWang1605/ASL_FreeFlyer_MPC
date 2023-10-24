"""
Script attempt to try mpc control
Joshua Lee (joshhlee@stanford.edu)
10/23/2023
"""

import numpy as np 
import random

class MPC_Controller:
    def __init__(self, state_observer, dynamics):
        self.observer = state_observer
        self._u = None
        self.goal_state = None
        self.command_history = np.zeros([20,8])
        self.command_seq = np.zeros(20)
        self.dynamics = dynamics

    def compute_dpos(self, state, goal):
        delta_pos = np.linalg.norm(goal[0:2] - state[0:2])
        delta_angle = goal[2] - state[2]

        return [delta_pos, delta_angle]

    def eval_input(self):
        min_cost = float('-inf')
        for i in range(10000):
            seq = np.zeros(20)
            for j in range(20):
                sample = random.randint(0,80) # samples in base 3
                eq[j] = sample_2binary(sample) # returns a list 20 lines long of random sequence of thrusters

            # compute the cost for each command

            for k in range(20):
                self.dynamics.integrate(seq[k], self.t, 1/self.SimFreq)
            predicted_state = self.dynamics.get_state()

            # distance from target, cost
            self.compute_dpos(predicted_state, self.goal_state)

            # gas cost
            gas_time = 0
            for m in seq:
                for n in m:
                    if n == 1:
                        gas_time+=1

            # switching cost

            # TODO: how to compute cost
            cost = 100000

            if cost < min_cost:
                min_cost = cost
                self.command_seq = seq
                
        return self.command_seq
    
    def get_input(self):
        return self.command_seq[0]

    # def eval_input(self):
    #     # we need to get the believed state and the actual state then we can control based on that

    #     self._u = np.array([[1,0,0,0,0,0,0,0]]).T
    #     return self._u

    # def get_input(self):
    #     return self._u

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
