import math
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt


def endurance(params):
    return -(math.exp(-2*(params[1]-math.sin(params[0]))**2)+math.sin(params[2]*params[3])+math.cos(params[4]*params[5]))

def endurance_p(x):
    n_particles = x.shape[0]
    j = [endurance(x[i]) for i in range(n_particles)]
    return np.array(j)


options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)
optimizer = ps.single.GlobalBestPSO(n_particles=1000, dimensions=6,
options=options, bounds=my_bounds)
optimizer.optimize(endurance_p, iters=100)

cost_history = optimizer.cost_history
# Perform optimization
plot_cost_history(cost_history)
plt.show()