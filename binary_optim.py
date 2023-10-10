"""
Script to perform MPC optimization for binary thruster commands for Free Flyer
Andrew Wang (aw1605@stanford.edu)
10/10/2023
"""

import numpy as np 
import casadi as ca 

class BinaryThrusterController:
    def __init__(self, state_observer, goal_state, numSteps = 5):
        self.observer = state_observer
        self.goal = goal_state
        self.N = numSteps

    def compute_input(self):
        curr_state = state_observer.get_state()

        # Evaluate optimal inputs
        opti = ca.Opti()
        u = opti.variable(8, self.N)

        opti.subject_to()

        totalGasCost = np.sum(u)
        switchCost = np.sum(np.abs(u[1:self.N,:] - u[0:self.N-1,:]))
        pathDistCost = 0
        for i in range(self.N):
            pathDistCost = pathDistCost +   


        cost = ca.mtimes()
        opti.minimize(cost)

        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)

        #solve optimization
        try:
            sol = opti.solve()
            uOpt = sol.value(u) #extract optimal input
            solverFailed = False
        except:
            print("Solver failed!")
            solverFailed = True
            uOpt = np.zeros((8, self.N)) #return a zero vector

        return uOpt[:,0]
        
