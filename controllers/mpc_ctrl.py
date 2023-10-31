"""
Script attempt to try mpc control
Joshua Lee (joshhlee@stanford.edu)
10/23/2023
"""

import numpy as np 
import random
import math

from dynamics import Dynamics, ThrusterDyn

DEPTH = 10

class MPC_Controller:
    def __init__(self, state_observer, dynamics):
        self.observer = state_observer
        self._u = None
        self.goal_state = np.array([3,3,0.5,0.5,0,0]).reshape(-1,1)
        # self.command_history = np.zeros([20,8])
        self.command_seq = None
        # self.dynamics = dynamics
        # self.sim = Dynamics(self.dynamics.get_state(), stateDim=6, inputDim=8)
        # self.CHECK = dynamics

    def compute_dstate(self, state, goal):
        delta_pos = np.linalg.norm(goal[0:2] - state[0:2]) # abs position difference
        delta_angle = min(abs(goal[2] - state[2]), abs(2*math.pi+state[2]-goal[2]))%math.pi # in radians
        delta_vel = np.linalg.norm(goal[3:5] - state[3:5])
        delta_angv = abs(state[5] - goal[5])

        return [delta_pos, delta_angle, delta_vel, delta_angv]

    def eval_input(self):
        min_cost = self.state_cost(self.observer.get_state())
        # min_cost = float('inf')
        for i in range(100):
            seq = np.zeros((DEPTH,8))
            # generates each of DEPTH number of commands for each sequence
            for j in range(DEPTH):
                sample = random.randint(0,80) 
                seq[j] = sample_2binary(sample)

            # TODO: how to compute cost ? 
            # print("sequence attmpted", seq)
            cost = self.cost_function(seq, ThrusterDyn(self.observer.get_state()))
            if cost < min_cost:
                min_cost = cost
                self.command_seq = seq
        if self.command_seq is not None:
            return self.command_seq[0]
        return np.zeros((DEPTH, 8))

    def get_input(self):
        if self.command_seq is not None:
            return self.command_seq[0]
        return np.zeros((8))
        # return self.command_seq[0]
    
    def cost_function(self, sequence, simulated):
        gas_cost = np.sum(sum(sequence))
        switch_cost = np.sum(abs(np.diff(sequence)))
        cost_state = 0
        time = 0
        for k in range(DEPTH):
                simulated.integrate(sequence[k], time, 1/1000)
                predicted_state = simulated.get_state()
                cost_state += self.state_cost(predicted_state)
                time += 0.001
        return (cost_state + gas_cost/1000 + switch_cost/1000)/DEPTH

    def state_cost(self, state):
        dist = self.compute_dstate(state, self.goal_state)
        pv_cost = abs(dist[0])**2   + abs(dist[2])
        ang_cost = (abs(dist[1])**2 + abs(dist[3]))
        # print(pv_cost,ang_cost)
        return pv_cost + ang_cost/20

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
