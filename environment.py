import numpy as np
import time

times = []

class Environment:
    def __init__(self, dynamics, controller, observer, goal, time=10, freq=10):
        """
        Initializes a simulation environment
        Args:
            dynamics (Dynamics): system dynamics object
            controller (Controller): system controller object
            observer (Observer): system state estimation object
        """
        #store system parameters
        self.dynamics = dynamics
        self.controller = controller
        self.observer = observer
        
        #define environment parameters
        self.iter = 0 #number of iterations
        self.t = 0 #time in seconds 
        self.done = False
        
        #Store system state
        self.x = self.dynamics.get_state() #Actual state of the system
        self.x0 = self.x #store initial condition for use in reset
        self.xObsv = None #state as read by the observer
        
        #Define simulation parameters
        self.SIM_FREQ = 1000 #integration frequency in Hz
        self.CONTROL_FREQ = freq #control frequency in Hz
        self.SIMS_PER_STEP = self.SIM_FREQ//self.CONTROL_FREQ
        self.TOTAL_SIM_TIME = time #total simulation time in s
        
        #Define history arrays
        self.xHist = np.zeros((self.dynamics.stateDimn, self.TOTAL_SIM_TIME*self.CONTROL_FREQ))
        self.uHist = np.zeros((self.dynamics.inputDimn, self.TOTAL_SIM_TIME*self.CONTROL_FREQ))
        self.tHist = np.zeros((1, self.TOTAL_SIM_TIME*self.CONTROL_FREQ))

        self.goal = goal
        
    def reset(self):
        """
        Reset the gym environment to its inital state.
        """
        #Reset gym environment parameters
        self.iter = 0 #number of iterations
        self.t = 0 #time in seconds
        self.done = False
        
        #Reset system state
        self.x = self.x0 #retrieves initial condiiton
        self.xObsv = None #reset observer state
        
        #Define history arrays
        self.xHist = np.zeros((self.dynamics.stateDimn, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))
        self.uHist = np.zeros((self.dynamics.inputDimn, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))
        self.tHist = np.zeros((1, self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1))

    def step(self):
        """
        Step the sim environment by one integration
        """
        prev_time = time.perf_counter()
        #retrieve current state information
        self._get_observation() #updates the observer
        
        #solve for the control input using the observed state
        # self.controller.eval_input(self.t)
        self.controller.eval_input()

        times.append(time.perf_counter() - prev_time)
        
        #Zero order hold over the controller frequency
        for i in range(self.SIMS_PER_STEP):
            self.dynamics.integrate(self.controller.get_input(), self.t, 1/self.SIM_FREQ) #integrate dynamics
            self.t += 1/self.SIM_FREQ #increment the time
            
        #update the deterministic system data, iterations, and history array
        self._update_data()        
    
    def _update_data(self):
        """
        Update history arrays and deterministic state data
        """
        #append the input, time, and state to their history queues
        self.xHist[:, self.iter] = self.x.reshape((self.dynamics.stateDimn, ))
        self.uHist[:, self.iter] = (self.controller.get_input()).reshape((self.dynamics.inputDimn, ))
        self.tHist[:, self.iter] = self.t
        
        #update the actual state of the system
        self.x = self.dynamics.get_state()
        
        #update the number of iterations of the step function
        self.iter +=1
    
    def _get_observation(self):
        """
        Updates self.xObsv using the observer data
        Useful for debugging state information.
        """
        self.xObsv = self.observer.get_state()
        # print("current orientation: ", self.observer.get_orient())
    
    # def _get_reward(self):
    #     """
    #     Calculate the total reward for ths system and update the reward parameter.
    #     Only implement for use in reinforcement learning.
    #     """
    #     return 0
    
    def _is_done(self):
        """
        Check if the simulation is complete
        Returns:
            boolean: whether or not the time has exceeded the total simulation time
        """
        #check current time with respect to simulation time
        # if self.t >= self.TOTAL_SIM_TIME:
        #     return True
        if self.iter >= self.xHist.shape[1]:
            return True
        return False
    
    def run(self, N = 1):
        """
        Function to run the simulation N times
        Inputs:
            N (int): number of simulation examples to run
        """
        #loop over an overall simulation N times
        for i in range(N):
            self.reset()
            while not self._is_done():
                print("Simulation Time Remaining: ", self.TOTAL_SIM_TIME - self.t)
                self.step() #step the environment while not done
            self.visualize() #render the result
            
    def visualize(self):
        """
        Provide visualization of the environment
        """
        print("Total Input Effort:", np.sum(np.trapz(self.uHist, self.tHist)))
        print("Total Input Effort:", np.sum(np.dot(self.uHist[:,:-1],np.diff(self.tHist).T)))
        # print("Total Input Effort:", np.sum(np.dot(self.uHist, np.diff(self.tHist).T)))
        print("Final Tracking Error:", self.xHist[:,-1] - self.goal.T)
        try:
            poserr_ind = np.argwhere(np.linalg.norm(self.xHist[:2,:] - self.goal[:2], axis=0) > 0.05).squeeze()
            therr_ind = np.argwhere(np.abs(self.xHist[2,:] - self.goal[2,0]) > 0.1)
            max_ind = np.max([poserr_ind[-1], therr_ind[-1]])
            print("Time settle to <0.05m positional and 0.1rad angular error", self.tHist[0, max_ind]) 
        except:
            print("Never converged")
        print("Average timestep time:", np.mean(times))
        self.dynamics.show_animation(self.xHist, self.uHist, self.tHist, freq=self.CONTROL_FREQ)