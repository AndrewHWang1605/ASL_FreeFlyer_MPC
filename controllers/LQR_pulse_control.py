"""
Script to supply constant control signal (for debugging)
Andrew Wang (aw1605@stanford.edu)
10/20/2023
"""

import numpy as np 

gain_f = 2.0
gain_df = 10.0
gain_t = 0.2
gain_dt = 0.4
K = np.array([[gain_f, 0, 0, gain_df, 0, 0],
                       [0, gain_f, 0, 0, gain_df, 0],
                       [0, 0, gain_t, 0, 0, gain_dt]])

class LQRPulseController:
    def __init__(self, state_observer, goal, Fmax = 0.2, r = 0.11461, K=K):
        self.observer = state_observer
        self.goal = goal
        self._u = None
        self.K = K
        self.Fmax = Fmax
        self.r = r

    def eval_input(self):
        state = self.observer.get_state()
        state_err = (self.goal - state)
        # state_err[2][0] = state_err[2][0] % (2*np.pi)
        state_err[2][0] = ((state_err[2][0] + np.pi) % (2 * np.pi)) - np.pi
        x, y, th, xd, yd, thd = state.squeeze()[0], state.squeeze()[1], state.squeeze()[2], state.squeeze()[3], state.squeeze()[4], state.squeeze()[5]
        
        if np.max(np.abs(state_err)) < 0.05:
            global_wrench = np.zeros((3,1))
        else:
            global_wrench = self.K @ state_err
        body_wrench = self._transform_wrench(global_wrench[:2,], th).squeeze()
        
        M = global_wrench.squeeze()[2]
        Fx, Fy = body_wrench[0], body_wrench[1]
        # print(body_wrench)
        self._u = self._map_to_thrusters(Fx, Fy, M)
        # print(self._u)
        return self._u

    def get_input(self):
        return self._u

    def _transform_wrench(self, glob_vec, th):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        return R.T @ glob_vec

    def _map_to_thrusters(self, Fx, Fy, M):
        print(Fx, Fy, M)
        cmd = np.zeros(8)
        u_Fx = np.clip(Fx / 2, -self.Fmax, self.Fmax)
        u_Fy = np.clip(Fy / 2, -self.Fmax, self.Fmax)
        if u_Fx > 0:
            cmd[1] = u_Fx
            cmd[4] = u_Fx
        else:
            cmd[0] = -u_Fx
            cmd[5] = -u_Fx
        if u_Fy > 0:
            cmd[3] = u_Fy
            cmd[6] = u_Fy
        else:
            cmd[7] = -u_Fy
            cmd[2] = -u_Fy

        u_M = np.clip(M / 4 / self.r, -self.Fmax, self.Fmax)
        if u_M > 0:
            cmd[[0, 2, 4, 6]] += u_M
        else:
            cmd[[1, 3, 5, 7]] += -u_M
        output = np.clip(cmd, 0, self.Fmax).reshape((8,1)) / self.Fmax
        # print(output)
        # output = np.array([1,0,1,0,1,0,1,0]).reshape((8,1))
        return output


