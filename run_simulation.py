#Our dependencies
from environment import *
from dynamics import *
from controller import *
from trajectory import *
from state_estimation import *
from lyapunov import *

#system boundary conditions
x0 = np.array([[0, 0, 0]]).T #Start Free Flyer at origin
xf = np.array([[1, 1, np.pi/2]]).T #Move free flyer to new location

#create a dynamics object for the double integrator
dynamics = Dynamics(x0)

#create an observer based on the dynamics object with noise parameters
mean = 0
sd = 0.01
observer = StateObserver(dynamics, mean, sd)

#create a planar quadrotor controller
controller = BinaryThrusterController(observer)

#create a simulation environment
env = Environment(dynamics, controller, observer)
env.reset()

#run the simulation
env.run()