"""
Script attempt to try mpc control
Joshua Lee (joshhlee@stanford.edu)
10/23/2023
"""

import numpy as np 
import random
import math

from dynamics import Dynamics, ThrusterDyn

DEPTH = 3
NUM_SAMPLES = 10
DISCOUNT = 0.75
dT = 0.25

class MPC_Controller:
    def __init__(self, state_observer, dynamics, xf):
        self.observer = state_observer
        self._u = None
        self.goal_state = xf
        self.prev_command = None
        self.total_sequences = np.zeros((DEPTH,8))
        self.previous = np.zeros((1,8))
        self.command_seq = None
        # self.dynamics = dynamics
        # self.sim = Dynamics(self.dynamics.get_state(), stateDim=6, inputDim=8)
        # self.CHECK = dynamics

    def compute_dstate(self, state, goal):
        '''
        Computes the distance from the target goal position.
        '''
        delta_pos = np.linalg.norm(goal[0:2] - state[0:2]) # abs position difference
        delta_angle = min(abs(goal[2] - state[2]), abs(2*math.pi+state[2]-goal[2]))%math.pi # in radians
        delta_vel = np.linalg.norm(goal[3:5] - state[3:5])
        delta_angv = abs(state[5] - goal[5])

        return [delta_pos, delta_angle, delta_vel, delta_angv]

    def get_input(self):
        '''
        Returns the last executed sequence 
        '''
        if self.command_seq is not None:
            return self.command_seq
        return np.zeros((8))
    
    def cost_function(self, sequence, simulated):
        '''
        cost of an entire sequence of commands. position, gas cost, switching cost.
        '''
        gas_cost = np.sum(sum(sequence))
        switch_cost = np.sum(abs(np.diff(sequence)))
        cost_state = 0
        time = 0
        for k in range(len(sequence)):
                simulated.integrate(sequence[k], time, dT)
                predicted_state = simulated.get_state()
                cost_state += self.state_cost(predicted_state)
                time += dT
        return (cost_state)/DEPTH

    def state_cost(self, state):
        dist = self.compute_dstate(state, self.goal_state)
        pv_cost = 100*abs(dist[0])**2 + abs(dist[2])
        # ang_cost = (abs(dist[1])**2 + abs(dist[3]))
        # print("pv_cost", pv_cost)
        # print("ang cost", ang_cost)
        return pv_cost  #+ ang_cost
    
    def generate_samples(self, D):
        # take 10 samples from the start generate a sample from the start

        seq = np.zeros((D,8)) # for a depth long sequence of commands
        for j in range(D):
            sample = random.randint(0,80) 
            seq[j] = sample_2binary(sample)
        return seq

    def monte_carlo(self, sequence, sim, depth):
        curr_cost = self.state_cost(self.observer.get_state())
        return 0


    def branch_bound(self, sequence, sim, depth):
        curr_cost = self.state_cost(self.observer.get_state()) # gets the current cost of that state that we are at. 

        optimal_sequence = []           # no idea what the best sequence next part of the sequence
        optimal_cost = curr_cost 

        # base case 
        if depth == 0:
            sim.integrate(sequence, 0, dT)
            predicted_state = sim.get_state()
            return self.state_cost(predicted_state), [sequence]
        
        sim.integrate(sequence, 0, dT)
        optimal_sequence.append(sequence)
        predicted_state = sim.get_state()

        sequences = off_by_one(sequence)
        sorted_list = []
        for seq_ in sequences:
            ''' 
            Iterate through the sequences and if they cost bad, then remove half of them. 
            '''
            new_sim = ThrusterDyn(sim.get_state())
            new_sim.integrate(seq_, 0, dT)

            predicted_state = new_sim.get_state()
            cost = self.state_cost(predicted_state)
            sorted_list.append((seq_, cost))
        shorter_seq = [s[0] for s in sorted_list[:3]]

        for seq in shorter_seq:
            
            result = self.branch_bound(seq, ThrusterDyn(predicted_state), depth-1)

            if result is not None:
                cost, sequence_taken = result
                if cost < optimal_cost:
                    optimal_cost = cost
                    optimal_sequence = sequence_taken
        if optimal_sequence is not None:
            return optimal_cost, [sequence] + optimal_sequence
        else:
            return None


    def recursive_costs(self, sequence, sim, depth, cost_prevstate):
        # we need to figure out what the cost at this current state is. That way we can try to make sure we converge to lower costs
        curr_cost = self.state_cost(self.observer.get_state())
        optimal_cost = float('inf')
        
        # base case
        if depth == DEPTH:
            sim.integrate(sequence, 0, dT)
            predicted_state = sim.get_state()
            return self.state_cost(predicted_state), [sequence]
        
        sim.integrate(sequence, depth, dT)
        predicted_state = sim.get_state()
        
        optimal_sequence = None

        sequences = off_by_one(sequence)
        for seq in sequences:
            result = self.recursive_costs(seq, ThrusterDyn(predicted_state), depth+1, 0)

            if result is not None:
                cost, sequence_taken = result
                if cost < optimal_cost:
                    optimal_cost = cost
                    optimal_sequence = sequence_taken
        if optimal_sequence is not None:
            return optimal_cost, [sequence] + optimal_sequence
        else:
            return None
        

    def eval_input(self):
        # go thru the information and return the best thruster action to take. 

        min_cost = float('inf')
        # we don't know what the cost is.
        
        for i in range(40):
            sim = ThrusterDyn(self.observer.get_state())
            number = random.randint(0,80)
            init_sample = sample_2binary(number)
        # by this step, we have a sample that we can find the best action resulting from it
            sample_result = self.branch_bound(init_sample, ThrusterDyn(self.observer.get_state()), DEPTH)
            
            if sample_result is not None:
                sample_cost, sample_sequence = sample_result
                if sample_cost < min_cost:
                    min_cost = sample_cost
                    self.command_seq = init_sample

        if self.command_seq is not None:
            self.prev_command = self.command_seq
            return self.command_seq
        return self.prev_command
    

def off_by_one(seq):
    '''
    Takes in a sequence and returns the 8 thruster commands that only differ by one
    thruster command.
    '''
    if seq is None:
        return None
    sequences = []
    # change the first one and assign to first in sequence
    for i in range(len(seq)):
        seq_ = seq.copy()
        seq_[i] = (seq_[i]+1)%2
        sequences.append(seq_)
    return np.array(sequences)

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
