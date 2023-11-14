#Our dependencies
import sys 
sys.path.append('./controllers')

from environment import *
from dynamics import *
from constant_control import ConstantController
from LQR_pulse_control import LQRPulseController
from Binarized_forceopt_control import BinarizedForceOptController
# from trajectory import *
from state_observer import *

#system boundary conditions
x0 = np.array([[0, 0, 0, 0, 0, 0]]).T #Start Free Flyer at origin
xf = np.array([[1, 1, np.pi/4, 0, 0, 0]]).T #Move free flyer to new location

#create a dynamics object for the double integrator
dynamics = ThrusterDyn(x0)

#create an observer based on the dynamics object with noise parameters
mean = 0
sd = 0.02
observer = StateObserver(dynamics, mean, sd)

freq = 5

#create a planar quadrotor controller
# controller = ConstantController(observer)
# controller = LQRPulseController(observer, xf)
controller = BinarizedForceOptController(observer, xf, freq)

# create a simulation environment
# env = Environment(dynamics, controller, observer, xf, time=30, freq=10) # Lqr config
env = Environment(dynamics, controller, observer, xf, time=20, freq=freq) # Mapped Continuous Force Opt config

env.reset()

#run the simulation
env.run()