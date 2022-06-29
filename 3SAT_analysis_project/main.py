import itertools
import random
import pygad
import numpy as np
import sys

sys.setrecursionlimit(10000)



def tcnfgen(m, k, horn=2):
    cnf = []
    def unique(l, k):
        t = random.randint(1, k)
        while(t in l):
            t = random.randint(1, k)
        return t
    r = random.randint(0, 1)
    for _ in range(m):
        x = unique([], k)
        y = unique([x], k)
        z = unique([x, y], k)
        if horn:
            cnf.append([(x, 1), (y, 0), (z, 0)])
        else:
            cnf.append([(x, r), (y, r()), (z, r())])
    return cnf


def convert(cnf):
    for k in range(len(cnf)):
        for l in range(len(cnf[k])):
            if cnf[k][l][1] == 1:
                cnf[k][l] = cnf[k][l][0]
            else:
                cnf[k][l] = -cnf[k][l][0]
    return cnf



def onlyLiterals(cnf):
    return [[y[0] for y in x] for x in cnf]


def onlyValues(cnf):
    return [[y[1] for y in x] for x in cnf]


# above code was used to test ways of solving SAT problems, however at the end I decided to use already existing files with SAT problems

def fromFile(filename):
    with open(filename) as f:
        lines = f.readlines()
        return [[int(x) for x in lines[x][:-len(lines[x]) + 2].split("  ")] for x in range(len(lines))]


def flatten_and_slice(cnf, size=1):
    slice = len(cnf) * size
    return [item for sublist in cnf[:slice] for item in sublist]


def convert_to_boolean(cnf):
    cnf_x_or_not_x = []
    for x in cnf:
        tmp = set()
        for y in x:
            if y > 0:
                tmp.add((y, True))
            else:
                tmp.add((-y, False))
        cnf_x_or_not_x.append(tmp)
    return cnf_x_or_not_x





def brute_force(cnf):
    cnf = convert_to_boolean(cnf)
    literals = set()
    for conj in cnf:
        for disj in conj:
            literals.add(disj[0])

    literals = list(literals)
    n = len(literals)
    for seq in itertools.product([True, False], repeat=n):
        a = set(zip(literals, seq))
        if all([bool(disj.intersection(a)) for disj in cnf]):
            return True, a

    return False, None


def select_literal(cnf):
    for c in cnf:
        for literal in c:
            return literal[0]


def dpll(cnf, assignments={}):
    if len(cnf) == 0:
        return True, assignments

    if any([len(c) == 0 for c in cnf]):
        return False, None

    l = select_literal(cnf)

    new_cnf = [c for c in cnf if (l, True) not in c]
    new_cnf = [c.difference({(l, False)}) for c in new_cnf]
    sat, vals = dpll(new_cnf, {**assignments, **{l: True}})
    if sat:
        return sat, vals

    new_cnf = [c for c in cnf if (l, False) not in c]
    new_cnf = [c.difference({(l, True)}) for c in new_cnf]
    sat, vals = dpll(new_cnf, {**assignments, **{l: False}})
    if sat:
        return sat, vals
    return False, None



def howManyLiterals(cnf):
    cnf = flatten_and_slice(cnf)
    numberOfLiterals = 0
    checked = []
    for k in cnf:
        if abs(k) not in checked:
            numberOfLiterals += 1
            checked.append(abs(k))
    return numberOfLiterals




import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
import time as t






cnfFile = fromFile('project/CNFs/S_100_403.txt')
cnfS = convert_to_boolean(cnfFile)


def fitness_smart(cnf, solution):
    cnfS = convert_to_boolean(cnf)
    hashMap = {}
    fitness = -(len(cnfS))
    cnf = flatten_and_slice(cnfS)
    literals = []
    for x in cnf:
        if x[0] not in literals:
            literals.append(x[0])
    for index in range(len(solution)):
        hashMap[literals[index]] = solution[index]

    for k in cnfS:
        for l in k:
            if l[1] == False and hashMap[l[0]] == 0:
                fitness += 1
                break
            elif l[1] == True and hashMap[l[0]] == 1:
                fitness += 1
                break

    return fitness


def wrapper_smart(solution, solution_idx):
    return fitness_smart(cnfFile, solution)


def fitness(cnf, solution):
    cnfFlat = flatten_and_slice(cnf)
    hashMap = {}
    fitness = 0
    for x in range(0, len(solution), 3):
        if 1 not in solution[x:x+3]:
            fitness -= 3
    for x in range(len(solution)):
        literal = cnfFlat[x]
        if -literal in hashMap.keys():
            if solution[x] == hashMap[-literal]:
                fitness -= 1
        elif literal in hashMap.keys():
            if solution[x] != hashMap[literal]:
                fitness -= 1
        else:
            hashMap[literal] = solution[x]
    return fitness

def wrapper(solution, solution_idx):
    return fitness(cnfFile, solution)




gene_space = [0, 1]
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=100,
                       mutation_type="random",
                       mutation_percent_genes = 100/howManyLiterals(cnfFile),
                    #    mutation_percent_genes=[(100/len(cnfFlat))*2, 100/len(cnfFlat)],
                       fitness_func=wrapper_smart,
                       crossover_type="single_point",
                       parent_selection_type="sss",
                       num_parents_mating=int(0.2 * 100),
                       sol_per_pop=200,
                       num_genes=howManyLiterals(cnfFile),

                       )

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()


print(solution, solution_fitness)

ga_instance.plot_fitness()



def swarm(x):
    n_particles = x.shape[0]
    j = [wrapper_smart(x[i], x) + 1 for i in range(n_particles)]
    return np.array(j)


options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':2, 'p':1}
optimizer = ps.discrete.BinaryPSO(n_particles=800, dimensions=howManyLiterals(cnfFile),
options=options)
result = optimizer.optimize(swarm, iters=100, verbose=False)

print(howManyLiterals(cnfFile))
print(cnfFile)
print(result)

cost_history = optimizer.cost_history
# Perform optimization
plot_cost_history(cost_history)
plt.show()





