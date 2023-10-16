"""
Free Flyer dynamics
"""
import math
import random
import numpy as np


# Return x(t+1) given x(t), u(t)

class Dynamics:

    def __init__(self, x0, stateDim=6, inputDim = 3):
        self.init = x0
        self.stateDim = stateDim
        self.inputDim = inputDim

        self._u = None

    def get_state(self):
        return self.init

    def derive(self, x, u, t):
        return np.zeros((self.stateDim,1))

    def integrate(self, u, t, dt):
        self._x = self.get_state() + self.derive(self.get_state(), u, t)*dt
        return self._x


class Dyn(Dynamics):

        # Thrusters Configuration
        #      (2) e_y (1)        ___
        #     <--   ^   -->      /   \
        #    ^  |   |   |  ^     v M  )
        # (3)|--o-------o--|(8)    __/
        #       | free- |
        #       | flyer |   ---> e_x
        #       | robot |
        # (4)|--o-------o--|(7)
        #    v  |       |  v
        #     <--       -->
        #      (5)     (6)

    def __init__(self, x0 = np.zeros((6,1)), m = 16, Ixx = 0.18, r = 0.1):
        self._m = m
        self._Ixx = Ixx
        self._r = r

    # with the position, return the velocity
    def derive(self, X, U, t):
        """
        Returns the derivative of the state vector
        Args:
            X => state variable, 6x1 numpy array at time t
            U => input array: 2x1 numpy array, Force, Moment
            t => time
        Returns:
            xDot: 6x1 x 1 derivative of the state vector
        """
        F = U[0:1,0]  # Force x and y
        M = U[2,0]    # Moment about z
        #unpack the state vector
        x_dot, y_dot = X[4, 0], X[5, 0] #velocities
        theta, theta_dot = X[2, 0], X[5, 0]             #orientations

        x_ddot, y_ddot = F[0,0]/self._m, F[1,0]/self._m
        theta_ddot = M/self._Ixx
        deriv = np.array([[x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot]])

        # Noise
        # for i in range(len(deriv)):
        #     deriv[i] = random.gauss(1, 0.05) * deriv[i]
        return deriv

def thrusters(index):
        """
        Returns the resultant force from the thruster specified
        Input:
            index: number thruster we want to get
        Returns:
            resultant (3x1) Force + moment ABOUT ROBOT AXIS from thruster    
        """
        # assert len(input) == 8, "input must be 8x1 mapping of thrusters"
        # assert len(output) == , "output gives "
        dim1 = 0.11461
        dim2 = 0.0955

        # gives all the thrusters positions relative to its center
        if index == 1: 
            tPos = np.array([dim2, dim1])
            F = np.array([0,-1])
        elif index == 2: 
            tPos = np.array([dim2, -dim1])
            F = np.array([0,1])
        elif index == 3:
            tPos = np.array([-dim1, dim2])
            F = np.array([0,-1])
        elif index == 4: 
            tPos = np.array([-dim1, -dim2])
            F = np.array([0,1])
        elif index == 5: 
            tPos = np.array([-dim2, -dim1])
            F = np.array([0,1])
        elif index == 6: 
            tPos = np.array([-dim2, dim1])
            F = np.array([0,-1])
        elif index == 7: 
            tPos = np.array([dim1, -dim2])
            F = np.array([0,1])
        elif index == 8: 
            tPos = np.array([dim1, dim2])
            F = np.array([0,-1])

        Moment = np.cross(tPos, F)
        Force = tPos[1]/(tPos[0]**2+tPos[1]**2)*np.array([-tPos[0], -tPos[1]])
        return np.append(Force, Moment).reshape(-1,1)
    
def resultant_force(input):
        """
        Returns the resultant total force and moment that we would predict
        Args:
            thruster command, (8x1), each corresponding to a thruster index
        Returns:
            Force and moment of the FF in its own frame. 
        """
        force_x = 0
        assert len(input) == 8, "check size of input, must have 8 binary values"
        for i in range(len(input)):
            # i gives the index in which I want to actuate -1. 
            if input[i] == 1:
                print("activated thruster", i+1)
                force_x += thrusters(i+1)
                # force_y += thrus
        return force_x
                
       