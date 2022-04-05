# import random as rd
# import pyswarms as ps
# from matplotlib import pyplot as plt
# from pyswarms.utils.functions import single_obj as fx
#
# # Set-up hyperparameters
# from pyswarms.utils.plotters import plot_cost_history
#
# options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
#
# # Call instance of GlobalBestPSO
# optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
#                                     options=options)
#
# cost_history = optimizer.cost_history
# # Perform optimization
# stats = optimizer.optimize(fx.sphere, iters=100)
# plot_cost_history(cost_history)
# pos_history = optimizer.pos_history
# print(pos_history)

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher

# Run optimizer
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
# historia kosztów i pozycji
cost_history, pos_history = optimizer.optimize(fx.sphere, iters=50)
# tworzenie animacji
m = Mesher(func=fx.sphere)
animation = plot_contour(pos_history=pos_history,
                         mesher=m,
                         mark=(0, 0))
animation.save('plot0.gif', writer='imagemagick', fps=10)



