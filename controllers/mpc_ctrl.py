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
        self.goal_state = np.array([1,1,0,0,0,0]).reshape(-1,1)
        self.command_history = np.zeros([20,8])
        self.command_seq = None
        self.dynamics = dynamics

    def compute_dpos(self, state, goal):
        delta_pos = np.linalg.norm(goal[0:2] - state[0:2])
        delta_angle = int(goal[2] - state[2])

        return [delta_pos, delta_angle]

    def eval_input(self):
        min_cost = float('inf')
        for i in range(100):
            seq = np.zeros((1,8))
            for j in range(1):
                sample = random.randint(1,80) # samples in base 3
                seq[j] = sample_2binary(sample) # returns a list 20 lines long of random sequence of thrusters
            time = 0
            # compute the cost for each command

            for k in range(1):
                self.dynamics.integrate(seq[k], time, 1/1000)
                time += 0.001

            predicted_state = self.dynamics.get_state()

            # distance from target, cost
            dist = self.compute_dpos(predicted_state, self.goal_state)

            # gas cost
            gas_cost = np.sum(sum(seq))

            # switching cost
            switch_cost = np.sum(abs(np.diff(seq)))

            # TODO: how to compute cost ? 
            cost = abs(dist[0])**2  + abs(dist[1])**2 + gas_cost + 0.1*switch_cost

            if cost < min_cost:
                print(dist)
                min_cost = cost
                self.command_seq = seq

        return self.command_seq[0]
    
    def get_input(self):
        return self.command_seq[0]

    # def eval_input(self):

    #     self._u = np.array([[1,0,0,0,0,0,0,0]]).T
    #     return self._u

    # def get_input(self):
    #     return self._u

def sample_2binary(num):
    '''
    Takes in random sample, converts to ternary numbers for each thruster pair,
    converts to binary for each thruster, then returns them as np array
    '''
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
    number = np.array([int(char) for char in binary_number])
    return number
