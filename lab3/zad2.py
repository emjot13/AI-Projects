import math
import pygad
import random as rd
import time as t

def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)

random_solutions = []
start_random = t.time()
for k in range(16000000):
    parameters = [rd.random() for x in range(6)]
    random_solutions.append(endurance(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]))
end_random = t.time()

bestRandomResult = max(random_solutions)

def fitness_func(solution, solution_idx):
    fitness = endurance(solution[0], solution[1], solution[2], solution[3], solution[4], solution[5])
    return fitness

ga_instance = pygad.GA(init_range_low=0,
                       init_range_high=1,
                       random_mutation_min_val=0,
                       random_mutation_max_val=1,
                       mutation_by_replacement=True,
                       num_generations=40000,
                       mutation_percent_genes = 30,
                       fitness_func=fitness_func,
                       crossover_type = "single_point",
                       parent_selection_type = "sss",
                       num_parents_mating = 5,
                       sol_per_pop = 15,
                       num_genes = 6

                        )


start_AI = t.time()
ga_instance.run()
end_AI = t.time()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution:\n{solution[0]}\n{solution[1]}\n{solution[2]}\n{solution[3]}\n{solution[4]}\n{solution[5]}\n")


print(f"Fitness value of the best solution = {solution_fitness}, elapsed time: {round(end_AI - start_AI, 2)} seconds.")
print(f"Best random solution: {bestRandomResult}, elapsed time {round(end_random - start_random, 2)} seconds.")
 
change_percent = ((float(solution_fitness)-bestRandomResult)/bestRandomResult)*100
print(f"Difference between AI and random solution: {solution_fitness - bestRandomResult}, percentage difference: {round(change_percent, 3)}%")