import math
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt


options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':2, 'p':1}


import pygad
import numpy

S = [1, 2, 3, 6, 10, 17, 25, 29, 30, 41, 51, 60, 70, 79, 80]

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#definiujemy funkcjÄ fitness
def fitness_func(solution):
    sum1 = numpy.sum(solution * S)
    solution_invert = 1 - solution
    sum2 = numpy.sum(solution_invert * S)
    fitness = numpy.abs(sum1-sum2)
    #lub: fitness = 1.0 / (1.0 + numpy.abs(sum1-sum2))
    return fitness


optimizer = ps.discrete.BinaryPSO(n_particles=10, dimensions=15,
options=options)
result = optimizer.optimize(fitness_func, iters=4000, verbose=False)

sum1 = 0
sum2 = 0
for k in range(len(result[1])):
    if result[1][k] == 0:
        sum1 += S[k]
    else:
        sum2 += S[k]

print(result[1], sum1, sum2, abs(sum2 - sum1))
cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()