"""
Free Flyer dynamics
"""
import math
import random
import numpy as np


# Return x(t+1) given x(t), u(t)

class Dynamics:

    def __init__(self, x0, stateDim=6, inputDim = 8):
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
        # (3)|--o-------o--|(0)    __/
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
        
    def set_wrench(self, wrench_body:Wrench2D):
        u_Fx = wrench_body_clipped.fx / (2 * self.p.actuators["F_max_per_thruster"])
        u_Fy = wrench_body_clipped.fy / (2 * self.p.actuators["F_max_per_thruster"])


    def set_body_wrench(self, wrench_body):
        # Clip the wrench
        wrench_body_clipped = self.clip_wrench(wrench_body)

        # Convert force
        u_Fx = wrench_body_clipped.fx / (2 * self.p.actuators.F_max_per_thruster)
        u_Fy = wrench_body_clipped.fy / (2 * self.p.actuators.F_max_per_thruster)
        if u_Fx > 0:
            u[2] = u_Fx
            u[5] = u_Fx
        else:
            u[1] = -u_Fx
            u[6] = -u_Fx
        if u_Fy > 0:
            u[4] = u_Fy
            u[7] = u_Fy
        else:
            u[0] = -u_Fy
            u[3] = -u_Fy

        # Convert torque
        u_M = wrench_body_clipped.tz / (
            4 * self.p.actuators.F_max_per_thruster * self.p.actuators.thrusters_lever_arm
        )
        if u_M > 0:
            for i in [1, 3, 5, 7]:
                u[i] += u_M
        else:
            for i in [0, 2, 4, 6]:
                u[i] += -u_M

        # Clip duty cycles to the range [0, 1]
        for i in range(8):
            u[i] = max(min(1.0, u[i]), 0.0)

        self.set_thrust_duty_cycle(u)

