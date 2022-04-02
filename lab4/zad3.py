import random as rd

import pyswarms
import pyswarms as ps
from matplotlib import pyplot as plt
from pyswarms.utils.functions import single_obj as fx
from pyswarms.backend.topology import Star



# Set-up hyperparameters
from pyswarms.utils.plotters import plot_cost_history
# from sympy.physics.units import dimensions
# from tests.optimizers.test_tolerance import n_particles

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Call instance of GlobalBestPSO
# optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
#                                     options=options)

optimizer = ps.single.GeneralOptimizerPSOBestPSO(n_particles=10, dimensions=2,
                                    options=options, topology=Star())

cost_history = optimizer.cost_history
# Perform optimization
stats = optimizer.optimize(fx.easom, iters=100)
plot_cost_history(cost_history)
plt.show()


# import pyswarms.backend as P
# from pyswarms.backend.swarms import Swarm
# from pyswarms.backend.topology import Star
#
# my_swarm = P.create_swarm(n_particles, dimensions)
# my_topology = Star()
#
# # Update best_cost and position
# Swarm.best_pos, Swarm.best_cost = my_topology.compute_gbest(my_swarm)


