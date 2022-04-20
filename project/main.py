# def fitness(individual, formula):
#     sum=0
#     for claus in range(len(formula)):
#         for element in range(3):

#             value=individual[abs(formula[claus][element])-1]

#             if formula[claus][element]<0:
#                 if value==0:
#                     value=1
#                     sum+=1
#                     break
#             elif formula[claus][element]>0:
#                 if value==1:
#                     sum+=1
#                     break

#     return sum

import random
import pygad
import numpy as np


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


def onlyLiterals(cnf):
    return [[y[0] for y in x] for x in cnf]


def onlyValues(cnf):
    return [[y[1] for y in x] for x in cnf]


someCNF = tcnfgen(10, 4)
onlyLiteral = onlyLiterals(someCNF)
onlyValue = list(np.concatenate(onlyValues(someCNF)).flat)


# print(someCNF, "\n", only)

def fromFile(filename):
    with open(filename) as f:
        lines = f.readlines()
        return [[int(x) for x in lines[x][:-3].split(" ")] for x in range(len(lines))]


cnf = fromFile('/home/LABPK/mjarzembinski/Pulpit/INF_Mateusz_Jarzembinski_275039/project/CNFs/S.txt')
cnfFlat = [item for sublist in cnf for item in sublist]
# print(cnfFlat)

# cnf = [[5, 3, 2], [-5, 2, 3], [-2, -3, 5]]


# solution = [1, 0, 1, 0, 1, 0, 0, 1, 1]

def fitness(solution, solution_idx):
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

# print(fitness(solution, 0))


gene_space = [0, 1]
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=500,
                       mutation_type="scramble",
                       mutation_percent_genes=100/len(cnfFlat),
                       fitness_func=fitness,
                       crossover_type="single_point",
                       parent_selection_type="sss",
                       num_parents_mating=int(0.2 * 1500),
                       sol_per_pop=1500,
                       num_genes=len(cnfFlat),

                       )

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()




print(solution, solution_fitness)

ga_instance.plot_fitness()
# print(someCNF)
# print(onlyValue)
# print(onlyLiteral)
# print(fitness(onlyValue, onlyLiteral, 0))
