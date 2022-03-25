import pygad
import random
import time as t
from statistics import mean


maze =      [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


def helper(solution, maze, current_point, end_point):
    for y in range(len(solution)):
        if solution[y] == 0:            # right
            current_point[1] += 1
        if solution[y] == 1:            # down
            current_point[0] += 1
        if solution[y] == 2:            # left
            current_point[1] -= 1 
        if solution[y] == 3:            # up
            current_point[0] -= 1
        fitness = -(abs(current_point[0] - end_point[0]) + abs(current_point[1] - end_point[1])) 
        if maze[current_point[0]][current_point[1]] == 1:
            return fitness 
        if fitness == 0:
            return fitness
    return fitness 



def fitness_func(solution, solution_idx):
    return helper(solution, maze, [1, 1], [10, 10]) 

# fitness_func([0, 0, 1, 3, 0, 3, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 3, 1, 0, 3, 0, 0, 1, 3], 10)


def optimize(solution):
    solution = list(solution)                   # looks for the first element in the solution that achieves the finish
    for x in range(len(solution)):
        if fitness_func(solution[:x+1], x) == 0:
            solution = solution[:x+1]

    pointlessMovesMap = [set([0, 2]), set([1, 3])]          # removes all pairs of moves (left, right and up, down) within the solution
    length = len(solution)
    x = 0
    while x < (length - 2):
            if set([solution[x], solution[x + 1]]) in pointlessMovesMap:
                del solution[x:x+2]
                length = len(solution)
            x += 1
    return solution

def convert(solution):
    result = ''
    directionsMap = {0: "right", 1: "down", 2: "left", 3: "up"}
    for index in range(len(solution)):
        if index == len(solution) - 1:
            result += directionsMap[solution[index]] + "."
            break
        result += directionsMap[solution[index]] + ", "            
    return result                


def randomSolutions(iterationsNum):
    randomResults = []
    for x in range(iterationsNum):
        randomResults.append(fitness_func([random.randint(0, 3) for y in range(30)], 0))
    return max(randomResults)




gene_space = [0, 1, 2, 3]
ga_instance = pygad.GA(gene_space = gene_space,                     
                       num_generations=200,
                       mutation_percent_genes = 10,
                       fitness_func=fitness_func,
                       crossover_type = "single_point",
                       parent_selection_type = "sss",
                       num_parents_mating = 50,
                       sol_per_pop = 150,
                       num_genes = 30,
                       
                        )

AItimes = []
for x in range(10):
    startAI = t.time()                        
    ga_instance.run()
    endAI = t.time()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if solution_fitness != 0:
        print("one of the solutions is not correcct")
        break
    result = endAI - startAI
    AItimes.append(result)


startRandom = t.time()
bestOfRandom = randomSolutions(100000)
endRandom = t.time()

print(AItimes)

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution (distance between final stopping point and destination): {solution_fitness}\n")
print(f"Best numerical entire solution:\n{solution}\nOrganism length: {len(solution)}\n")
print(f"More readable and optimized to only neccesary steps solution:\n{convert(optimize(solution))}\nOptimized length: {len(optimize(solution))}\n")
print("Comparison between finding paths randomly and with genetic algorithm")
print(f"Parameters: distance between the destination and the best solution, average elapsed time from 10 solutions:\n\nGenetic algorithm: {solution_fitness}, {mean(AItimes)} seconds")
print(f"Random: {bestOfRandom}, {endRandom - startRandom} seconds")









