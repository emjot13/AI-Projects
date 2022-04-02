import random as rd
import pyswarms as ps
from matplotlib import pyplot as plt
from pyswarms.utils.functions import single_obj as fx

# Set-up hyperparameters
from pyswarms.utils.plotters import plot_cost_history

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Call instance of GlobalBestPSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                    options=options)

cost_history = optimizer.cost_history
# Perform optimization
stats = optimizer.optimize(fx.sphere, iters=100)
plot_cost_history(cost_history)
plt.show()