"""
Script to supply constant control signal (for debugging)
Andrew Wang (aw1605@stanford.edu)
10/20/2023
"""

import numpy as np 
from casaid import *



class BinaryContOptController:
    def __init__(self, state_observer, goal, Fmax = 0.2):
        self.observer = state_observer
        self.goal = DM(goal)
        self._u = None
        self.Fmax = Fmax

        dim1 = 0.11461
        dim2 = 0.0955

        self.thruster_orien = np.array(([-1,0],[1,0],[0,-1],[0,1],[1,0],[-1,0],[0,1],[0,-1]))
        self.thruster_pos = np.array(([dim2, dim1],[-dim2, dim1],[-dim1, dim2],[-dim1, -dim2],
                                     [-dim2, -dim1],[dim2, -dim1],[dim1, -dim2],[dim1, dim2]))
        self.m = 16
        self.Ixx = 0.18
        # Initialize constraints
        # Bounds on x (x,y,th,xdot,ydot,thdot)
        self.lbx = [-4., -4., -np.pi, -1., -1., -1.]
        self.ubx = [4., 4., np.pi, 1., 1., 1.]
        self.lbu = [0., 0., 0., 0., 0., 0., 0., 0.] 
        self.ubu = [1., 1., 1., 1., 1., 1., 1., 1.]

        

    def eval_input(self):
        x0 = self.observer.get_state()
        T0 = 2
        N = 50

        # Declare model variables
        x = MX.sym('x', 6) # x,y,th,xdot,ydot,thdot
        u = MX.sym('u', 8) # 8 binary thrusters
        T = MX.sym('T') # Time

        body_Fx, body_Fy, M = 0, 0, 0

        for i in range(8):
            f = u[i]*self.thruster_orien[i]
            r = self.thruster_pos[i]
            body_Fx += f[0]
            body_Fy += f[1]
            M += f[0]*r[1] - f[1]*r[0]

        world_Fx = body_Fx*cos(th) - body_Fy*sin(th)
        world_Fy = body_Fx*sin(th) + body_Fy*cos(th)
        
        # Model equations
        xdot = vertcat(x[3],
                       x[4],
                       x[5],
                       world_Fx / self.m,
                       world_Fy / self.m,
                       M / self.Ixx)

        k_t = 1
        k_pos = 2
        k_input = 1
        k_velo = 3
        # Stepwise Cost
        L =  k_pos*(normsq(x[0:2]-goal[0:2])) + k_input*(normsq(u))

        if True:
            # Fixed step Runge-Kutta 4 integrator
            M = 4 # RK4 steps per interval
            DT = T/N/M
            f = Function('f', [x, u], [xdot, L])    # Define function to take in current state/input and output xdot and cost
            X0 = MX.sym('X0', 4)                
            U = MX.sym('U', 4)
            X = X0
            Q = 0
            for j in range(M):
                k1, k1_q = f(X, U)
                k2, k2_q = f(X + DT/2 * k1, U)
                k3, k3_q = f(X + DT/2 * k2, U)
                k4, k4_q = f(X + DT * k3, U)
                X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
                Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
            F = Function('F', [T, X0, U], [X, Q],['time', 'x0','p'],['xf','qf'])   # Take in initial state and current input and outputs final state and cost after one step (Runge-Kutta integrated)

        # Initial guess for u
        u_start = [DM([0,0.,0.,0.])] * N

        # Get a feasible trajectory as an initial guess
        xk = DM(x0)
        T_start = DM(T0)
        x_start = [xk]
        for k in range(N):
            xk = F(time=T_start, x0=xk, p=u_start[k])['xf']
            x_start += [xk]

        # Start with an empty NLP
        w=[]
        w0 = []
        lbw = []
        ubw = []
        discrete = []
        # discrete2 = []
        J = 0
        g=[]
        lbg = []
        ubg = []

        # "Lift" initial conditions
        X0 = MX.sym('X0', 4)
        w += [X0]
        lbw += x0
        ubw += x0
        w0 += [x_start[0]]
        discrete += [False, False, False, False]
        # discrete2 += [False, False, False, False]


        # Formulate the NLP
        Xk = X0
        for k in range(N):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), 4)
            w   += [Uk]
            lbw += lbu
            ubw += ubu
            w0  += [u_start[k]]
            discrete += [True, True, True, True]

            # Integrate till the end of the interval
            Fk = F(time=T, x0=Xk, p=Uk)
            Xk_end = Fk['xf']
            J=J+Fk['qf']

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k+1), 4)
            w   += [Xk]
            lbw += lbx
            ubw += ubx
            w0  += [x_start[k+1]]
            discrete += [False, False, False, False]

            # Add equality constraint
            g   += [Xk_end-Xk]
            lbg += [0, 0, 0, 0]
            ubg += [0, 0, 0, 0]


        w   += [T]
        lbw += [0]
        ubw += [5]
        w0  += [T0]
        discrete += [False]
        J = J + k_t*T
        J = J + k_velo*normsq(Xk_end[2:])+ 10*k_pos*(normsq(Xk_end[0:2]-goal[0:2]))

        # Concatenate decision variables and constraint terms
        w = vertcat(*w)
        g = vertcat(*g)

        # Create an NLP solver
        nlp_prob = {'f': J, 'x': w, 'g': g}
        nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, {"discrete": discrete})
        # nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete})
        cont_nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); # Solve relaxed problem


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
        output = np.clip(cmd, -self.Fmax, self.Fmax).reshape((8,1)) / self.Fmax
        # print(output)
        # output = np.array([1,0,1,0,1,0,1,0]).reshape((8,1))
        return output


