"""
Free Flyer dynamics
"""
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches


# Return x(t+1) given x(t), u(t)

class Dynamics:

    def __init__(self, x0, stateDim=6, inputDim=8):
        self.init = x0
        self._x = x0
        self.stateDimn = stateDim
        self.inputDimn = inputDim

        self._u = None

    def get_state(self):
        return self._x

    def derive(self, x, u, t):
        return np.zeros((self.stateDimn,1))

    def integrate(self, u, t, dt):
        self._x = self.get_state() + self.derive(self.get_state(), u, t)*dt
        return self._x


class ThrusterDyn(Dynamics):

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
        super().__init__(x0, stateDim=6, inputDim = 8)
        self._m = m
        self._Ixx = Ixx
        self._r = r

    # with the position, return the velocity
    def derive(self, X, U, t):
        """
        Returns the derivative of the state vector
        Args:
            X => state variable, 6x1 numpy array at time t
            U => input array: 8x1 numpy array, 8 thrusters
            t => time
        Returns:
            xDot: 6x1 x 1 derivative of the state vector
        """
        #unpack the state vector
        x_dot, y_dot = X[3, 0], X[4, 0] #velocities
        theta, theta_dot = X[2, 0], X[5, 0]             #orientations

        res = self.resultant_force_and_moment(U)
        Fbody = res[0:2]  # Force x and y
        F = self.get_rotmatrix_body_to_world(theta) @ Fbody

        M = res[2].squeeze()    # Moment about z

        x_ddot, y_ddot = F[0,0]/self._m, F[1,0]/self._m
        theta_ddot = M/self._Ixx
        deriv = np.array([[x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot]]).T

        # Noise
        # for i in range(len(deriv)):
        #     deriv[i] = random.gauss(1, 0.05) * deriv[i]
        return deriv

    def thrusters(self, index):
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
            """
            Returns the resultant force from the thruster specified
            Input:
                index: number thruster we want to get
            Returns:
                resultant (3x1) Force + moment ABOUT ROBOT AXIS from thruster    
            """
            assert index <= 8 and index >= 1, "input must be 8x1 mapping of thrusters"
            # assert len(output) == , "output gives "
            dim1 = 0.11461
            dim2 = 0.0955
            Fmax = 0.2

            # gives all the thrusters positions relative to its center
            if index == 1: 
                tPos = np.array([dim2, dim1])
                # F = np.array([0,-1])
                F = np.array([-1,0])
            elif index == 2: 
                # tPos = np.array([dim2, -dim1])
                # F = np.array([0,1])
                tPos = np.array([-dim2, dim1])
                F = np.array([1,0])
            elif index == 3:
                tPos = np.array([-dim1, dim2])
                F = np.array([0,-1])
            elif index == 4: 
                tPos = np.array([-dim1, -dim2])
                F = np.array([0,1])
            elif index == 5: 
                tPos = np.array([-dim2, -dim1])
                # F = np.array([0,1])
                F = np.array([1,0])
            elif index == 6: 
                # tPos = np.array([-dim2, dim1])
                # F = np.array([0,-1])
                tPos = np.array([dim2, -dim1])
                F = np.array([-1,0])
            elif index == 7: 
                # tPos = np.array([dim1, -dim2])
                tPos = np.array([dim1, -dim2])
                F = np.array([0,1])
            elif index == 8: 
                tPos = np.array([dim1, dim2])
                F = np.array([0,-1])

            Moment = np.cross(tPos, F*Fmax)
            # Force = tPos[1]/(tPos[0]**2+tPos[1]**2)*np.array([-tPos[0], -tPos[1]]) * Fmax
            Force = F*Fmax # Not too sure about this one
            return np.append(Force, Moment).reshape(-1,1)
        
    def resultant_force_and_moment(self, input):
            """
            Returns the resultant total force and moment that we would predict
            Args:
                thruster command, (8x1), each corresponding to a thruster index
            Returns:
                Force and moment of the FF in its own frame. 
            """
            force_x = 0
            squeezed_input = input.squeeze()
            assert len(input) == 8, "check size of input, must have 8 binary values"
            for i in range(len(squeezed_input)):
                # i gives the index in which I want to actuate -1. 
                if squeezed_input[i] == 1:
                    # print("activated thruster", i+1)
                    force_x += self.thrusters(i+1)
            return force_x
                    
    def get_rotmatrix_body_to_world(self, theta):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return R

    def show_animation(self, xData, uData, tData, animate = True):
        """
        Shows the animation and visualization of data for this system.
        Args:
            xData (stateDimn x N Numpy array): state vector history array
            u (inputDimn x N numpy array): input vector history array
            t (1 x N numpy array): time history
            animate (bool, optional): Whether to generate animation or not. Defaults to True.
        """
        #Set constant animtion parameters
        GOAL_POS = [1, 2]
        FREQ = 50 #control frequency, same as data update frequency
        L = 0.15 #Forward arrow length
        
        if animate:
            fig, ax = plt.subplots()
            # set the axes limits
            ax.axis([-3, 3, -3, 3])
            # set equal aspect such that the circle is not shown as ellipse
            ax.set_aspect("equal")
            # create a point in the axes
            # point, = ax.plot(0,1, marker="o")
            circ = patches.Circle((0,0), 0.11, color='blue')
            circle = ax.add_patch(circ)
            num_frames = xData.shape[1]-1

            #define the line for the quadrotor
            line, = ax.plot([], [], '-r', lw=2)
            
            #plot the goal position
            # ax.scatter([GOAL_POS[0]], [GOAL_POS[1]], color = 'y')
                
            def animate(i):
                x = xData[0, i]
                y = xData[1, i]
                circle.center = (x,y)
                
                #draw the quadrotor line body
                theta = xData[2, i]
                x1 = x
                x2 = x + L*np.cos(theta)
                y1 = y 
                y2 = y + L*np.sin(theta)
                thisx = [x1, x2]
                thisy = [y1, y2]
                line.set_data(thisx, thisy)
                
                return line, circle
            
            anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1/FREQ*1000, blit=True)
            plt.xlabel("X Position (m)")
            plt.ylabel("Y Position (m)")
            plt.title("Position of Free Flyer")
            plt.show()
            
        #Plot each state variable in time
        fig, axs = plt.subplots(6)
        fig.suptitle('Evolution of States in Time')
        xlabel = 'Time (s)'
        ylabels = ['X Pos (m)', 'Y Pos (m)', 'Theta (rad)', 'X Vel (m/s)', 'Y Vel (m/s)', 'Angular Vel (rad/s)']
        # goalStates = [0, 1, 2, 0, 0, 0, 0, 0]
        #plot the states
        for i in range(6):
            axs[i].plot(tData.reshape((tData.shape[1], )).tolist(), xData[i, :].tolist())
            #plot the goal state for each
            # axs[n].plot(tData.reshape((tData.shape[1], )).tolist(), [goalStates[i]]*tData.shape[1], 'r:')
            axs[i].set(ylabel=ylabels[i]) #pull labels from the list above
            axs[i].grid()
        axs[5].set(xlabel = xlabel)
        plt.show()
        #plot the inputs in a new plot
        fig, axs = plt.subplots(8)
        fig.suptitle('Evolution of Inputs in Time')
        xlabel = 'Time (s)'
        ylabels = [1,2,3,4,5,6,7,8]
        for i in range(len(ylabels)):
            axs[i].plot(tData.reshape((tData.shape[1], )).tolist(), uData[i, :].tolist())
            axs[i].set(ylabel=ylabels[i])
            axs[i].grid()
        axs[1].set(xlabel = xlabel)
        plt.show()