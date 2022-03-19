import numpy
import pygad


#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

S = ["clock", "landscape picture", "portrait", "radio", "laptop", "bedside lamp", "silver cutlery", "china", "bronze figure", "leather purse", "hoover"]
prices = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]
weights = [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]



#definiujemy funkcjÄ fitness
def fitness_func(solution, solution_idx):
    pricesToCheck = numpy.sum(solution * prices)
    weightToCheck = numpy.sum(solution * weights)
    if weightToCheck > 25:
        fitness = 0
        return fitness
    fitness = pricesToCheck
    return fitness

fitness_function = fitness_func

#ile chromsomĂłw w populacji
#ile genow ma chromosom
sol_per_pop = 10
num_genes = len(S)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 30
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 10

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

#uruchomienie algorytmu
ga_instance.run()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
prediction = numpy.sum(prices*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

items = [S[k] for k in range(len(solution)) if solution[k] == 1.0]
print(f"The most valuable set of items is: {', '.join(map(str, items))} worth in total {int(solution_fitness)}.")

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()