#Our dependencies
import sys 
sys.path.append('./controllers')

from environment import *
from dynamics import *
from constant_control import ConstantController
from mpc_ctrl import MPC_Controller
# from trajectory import *
from state_observer import *

#system boundary conditions
x0 = np.array([[0, 0, 0, 0, 0, 0]]).T #Start Free Flyer at origin
xf = np.array([[1, 1, np.pi/2, 0, 0, 0]]).T #Move free flyer to new location

#create a dynamics object for the double integrator
dynamics = ThrusterDyn(x0)
sim = ThrusterDyn(x0)

#create an observer based on the dynamics object with noise parameters
mean = 0
sd = 0.00
observer = StateObserver(dynamics, mean, sd)

#create a planar quadrotor controller
controller = ConstantController(observer)
# controller = MPC_Controller(observer, dynamics) # if mpc

#create a simulation environment
env = Environment(dynamics, controller, observer)
env.reset()

#run the simulation
env.run()