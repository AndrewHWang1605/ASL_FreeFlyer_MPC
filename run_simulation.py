#Our dependencies
import sys 
sys.path.append('./controllers')

from environment import *
from dynamics import *
from constant_control import ConstantController
from LQR_pulse_control import LQRPulseController
from Binarized_forceopt_control import BinarizedForceOptController
from Binarized_thrusteropt_control_clean import BinarizedThrustOptController
from MixedInt_thrusteropt_control import MixedIntThrusterOptController
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

freq = 1

#define controller
# controller = ConstantController(observer)
# controller = LQRPulseController(observer, xf)
# controller = BinarizedForceOptController(observer, xf, 3)
controller = BinarizedThrustOptController(observer, xf, 3)
# controller = MixedIntThrusterOptController(observer, xf, freq)


# create a simulation environment
# env = Environment(dynamics, controller, observer, xf, time=30, freq=10) # Lqr config
# env = Environment(dynamics, controller, observer, xf, time=30, freq=3) # Mapped Continuous Force Opt config
env = Environment(dynamics, controller, observer, xf, time=30, freq=3) # Continuous Thruster Opt config
# env = Environment(dynamics, controller, observer, xf, time=15, freq=freq) # Mixed-Int Thruster Opt config


env.reset()

#run the simulation
env.run()