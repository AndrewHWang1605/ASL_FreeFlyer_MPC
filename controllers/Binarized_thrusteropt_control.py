"""
Script with binarized thruster output w/optimization-based controller
Andrew Wang (aw1605@stanford.edu)
01/11/2024
"""

import numpy as np 
from casadi import *
import matplotlib.pyplot as plt


class BinarizedThrustOptController:
    def __init__(self, state_observer, goal, freq):
        self.observer = state_observer
        self.goal = DM(list(goal))
        self._u = None
        
        self.freq = freq

        dim1 = 0.11461
        dim2 = 0.0955
        self.r = dim1

        self.thruster_orien = np.array(([-1,0],[1,0],[0,-1],[0,1],[1,0],[-1,0],[0,1],[0,-1]))
        self.thruster_pos = np.array(([dim2, dim1],[-dim2, dim1],[-dim1, dim2],[-dim1, -dim2],
                                     [-dim2, -dim1],[dim2, -dim1],[dim1, -dim2],[dim1, dim2]))
        self.m = 16
        self.Ixx = 0.18
        self.Fmax = 0.2
        # Initialize constraints on x (x,y,th,xdot,ydot,thdot)
        self.lbx = [-4., -4., -np.pi, -0.5, -0.5, -0.5]
        self.ubx = [4., 4., np.pi, 0.5, 0.5, 0.5]
        self.lbu = [-self.Fmax] * 4 
        self.ubu = [self.Fmax] * 4 

        self.last_input_seq = None

        ########## Initialize solver ##########
        self.w0, self.cont_nlp_solver, self.lbw, self.ubw, self.lbg, self.ubg = self.init_solver()
        
    def init_solver(self):
        T = 3
        N = int(T * self.freq)
        print(list(self.observer.get_state()))
        x0 = list(self.observer.get_state())

        # Declare model variables
        x = MX.sym('x', 6) # x,y,th,xdot,ydot,thdot
        u = MX.sym('u', 4) # 4 trinary thrusters (-1, 0, 1)
        
        body_Fx, body_Fy, M = self._map_to_force(u)

        th = x[2]
        world_Fx = body_Fx*cos(th) - body_Fy*sin(th)
        world_Fy = body_Fx*sin(th) + body_Fy*cos(th)
        
        # Model equations
        xdot = vertcat(x[3],
                       x[4],
                       x[5],
                       world_Fx / self.m,
                       world_Fy / self.m,
                       M / self.Ixx)

        # k_t = 0.2
        k_th = 1
        k_pos = 2.5
        k_input = 3
        k_velo = 40
        # Stepwise Cost
        L =  k_th*(self.goal[2]-x[2])**2 + k_pos*(self.normsq(self.goal[0:2] - x[0:2])) + k_input*(self.normsq(u)) + k_velo*self.normsq(x[3:])

        # Fixed step Runge-Kutta 4 integrator
        M = 4 # RK4 steps per interval
        DT = T/N/M
        f = Function('f', [x, u], [xdot, L])    # Define function to take in current state/input and output xdot and cost
        X0 = MX.sym('X0', 6)                
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
        F = Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])   # Take in initial state and current input and outputs final state and cost after one step (Runge-Kutta integrated)

        # Initial guess for u
        u_start = [DM([0.,0.,0.,0.])] * N

        # Get a feasible trajectory as an initial guess
        xk = DM(x0)
        x_start = [xk]
        for k in range(N):
            xk = F(x0=xk, p=u_start[k])['xf']
            x_start += [xk]

        # Start with an empty NLP
        w=[]
        w0 = []
        lbw = []
        ubw = []
        discrete = []
        J = 0
        g=[]
        lbg = []
        ubg = []

        # "Lift" initial conditions
        X0 = MX.sym('X0', 6)
        w += [X0]
        lbw += x0
        ubw += x0
        w0 += [x_start[0]]

        # Formulate the NLP
        Xk = X0
        for k in range(N):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), 4)
            w   += [Uk]
            lbw += self.lbu
            ubw += self.ubu
            w0  += [u_start[k]]

            # Integrate till the end of the interval
            Fk = F(x0=Xk, p=Uk)
            Xk_end = Fk['xf']
            J=J+Fk['qf']

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k+1), 6)
            w   += [Xk]
            lbw += self.lbx
            ubw += self.ubx
            w0  += [x_start[k+1]]
            # Add equality constraint
            g   += [Xk_end-Xk]
            lbg += [0, 0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0, 0]

        J = J + 10*k_pos*(self.normsq(Xk_end[0:2]-self.goal[0:2])) + 10*k_th*(self.normsq(self.goal[2]-Xk_end[2])) + 10*k_velo*self.normsq(Xk_end[3:])

        # Concatenate decision variables and constraint terms
        w = vertcat(*w)
        g = vertcat(*g)

        # Create an NLP solver
        nlp_prob = {'f': J, 'x': w, 'g': g}
        # nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, {"discrete": discrete})
        # nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete})
        cont_nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); # Solve relaxed problem
        return w0, cont_nlp_solver, lbw, ubw, lbg, ubg
    
    def eval_input(self):
        x0 = list(self.observer.get_state())

        self.lbw[:len(x0)] = x0
        self.ubw[:len(x0)] = x0
        self.w0[0] = x0
        
        sol = self.cont_nlp_solver(x0=vertcat(*self.w0), lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        output = sol['x']
        print(output)
        u0_opt, u1_opt, u2_opt, u3_opt, u4_opt, u5_opt, u6_opt, u7_opt = self.unpack_wopt(output)
        cont_thrust = np.array([u0_opt[0], u1_opt[0], u2_opt[0], u3_opt[0], u4_opt[0], u5_opt[0], u6_opt[0], u7_opt[0]]).reshape((8,1)) / self.Fmax
        self._u = cont_thrust > 0.5

        # Warm Start next run
        self.w0 = self.get_next_warm_start(output)
        
        def plot_sol(x_opt, y_opt, th_opt):
            # tgrid = [T_opt/N*k for k in range(N+1)]
            plt.figure(figsize=(10,8))
            plt.subplot(5,1,1)
            plt.plot(x_opt, y_opt,'*-')

            plt.subplot(5,1,2)
            plt.plot(x_opt, 'k-')
            # plt.plot([0,T], [goal[0].__float__(),goal[0].__float__()], 'k--')
            # plt.plot(tgrid, y_opt, 'r-')
            # plt.plot([0,T], [goal[1].__float__(),goal[1].__float__()], 'r--')
            # plt.legend(['x','xgoal','y','ygoal'])

            plt.subplot(5,1,3)
            plt.plot(y_opt, 'k-')
            # plt.plot(tgrid, vertcat(DM.nan(1), fx_opt))
            # # plt.step(tgrid, vertcat(DM.nan(1), u0_opt), 'r-')
            # # plt.step(tgrid, vertcat(DM.nan(1), u1_opt), 'b-')
            # plt.ylabel("X thrusters")
            # plt.grid(True)

            # plt.subplot(5,1,4)
            # plt.plot(tgrid, vertcat(DM.nan(1),fy_opt))
            # # plt.step(tgrid, vertcat(DM.nan(1), u2_opt), 'r-')
            # # plt.step(tgrid, vertcat(DM.nan(1), u3_opt), 'b-')
            # plt.xlabel('t')
            # plt.ylabel("Y thrusters")

            # plt.subplot(5,1,5)
            # plt.plot(tgrid, vertcat(DM.nan(1),m_opt))
            # # plt.step(tgrid, vertcat(DM.nan(1), u2_opt), 'r-')
            # # plt.step(tgrid, vertcat(DM.nan(1), u3_opt), 'b-')
            # plt.xlabel('t')
            # plt.ylabel("Moment")
            # plt.grid(True)

            # plt.suptitle(x_opt[:,0])
            plt.show()

        # plot_sol(x_opt, y_opt, th_opt)
        # sys.exit()
        return self._u

    
    def get_input(self):
        return self._u

    def _transform_glob_to_bod_wrench(self, glob_vec, th):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        return R.T @ glob_vec

    def _transform_bod_to_glob_wrench(self, bod_vec, th):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        return R @ bod_vec

    def _map_to_force(self, u):
        # Compute body-frame force from thrusters
    
        Fx = -u[0] + u[2] #-u[0] + u[1] - u[5] + u[4]
        Fy = -u[1] + u[3] #-u[2] + u[3] - u[7] + u[6]
        M = self.r * (u[0]+u[1]+u[2]+u[3]) #-self.r * (u[1]+u[3]+u[5]+u[7]) + self.r * (u[0]+u[2]+u[4]+u[6])
        return Fx, Fy, M

    def normsq(self, x):
        sum = 0
        for i in range(x.shape[0]):
            sum += x[i]**2
        return sum

    def unpack_wopt(self, w_opt):
        w_opt = w_opt.full().flatten()
        # x_opt = w_opt[0::10]
        # y_opt = w_opt[1::10]
        # th_opt = w_opt[2::10]
        # xdot_opt = w_opt[3::10]
        # ydot_opt = w_opt[4::10]
        # thdot_opt = w_opt[5::10]
        u0_opt = np.multiply(w_opt[6::10] > 0, w_opt[6::10])
        u1_opt = np.multiply(w_opt[6::10] < 0, -w_opt[6::10])
        u2_opt = np.multiply(w_opt[7::10] > 0, w_opt[7::10])
        u3_opt = np.multiply(w_opt[7::10] < 0, -w_opt[7::10])
        u4_opt = np.multiply(w_opt[8::10] > 0, w_opt[8::10])
        u5_opt = np.multiply(w_opt[8::10] < 0, -w_opt[8::10])
        u6_opt = np.multiply(w_opt[9::10] > 0, w_opt[9::10])
        u7_opt = np.multiply(w_opt[9::10] < 0, -w_opt[9::10])
        return u0_opt, u1_opt, u2_opt, u3_opt, u4_opt, u5_opt, u6_opt, u7_opt #, x_opt, y_opt, th_opt, xdot_opt, ydot_opt, thdot_opt

    def get_next_warm_start(self, w_opt):
        output = w_opt.full().flatten()
        x = [output[i:6+i] for i in range(w_opt.size()[0]//10+1)]
        u = [output[6+i:10+i] for i in range(w_opt.size()[0]//10)]

        w0 = [DM(x[1])]
        for i in range(len(u)-1):
            w0 += [DM(u[i+1])]
            w0 += [DM(x[i+2])]
        w0 += [DM(u[-1])]
        w0 += [DM(x[-1])]

        return w0
