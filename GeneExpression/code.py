import numpy as np

def fitness_function(x):
    return sum(x**2)

def initialize_population(pop_size, gene_length, bounds):
    lb, ub = bounds
    return lb + (ub - lb) * np.random.rand(pop_size, gene_length)

def selection(pop, fitness):
    idx = np.argsort(fitness)
    return pop[idx[:len(pop)//2]]

def crossover(parents, pop_size):
    offspring = []
    while len(offspring) < pop_size:
        p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
        point = np.random.randint(1, len(p1))
        child = np.concatenate([p1[:point], p2[point:]])
        offspring.append(child)
    return np.array(offspring)

def mutation(pop, mutation_rate, bounds):
    lb, ub = bounds
    for i in range(len(pop)):
        if np.random.rand() < mutation_rate:
            g = np.random.randint(len(pop[i]))
            pop[i][g] = lb + (ub - lb) * np.random.rand()
    return pop

def gene_expression(pop):
    return pop  

def gea(fitness_function, pop_size=10, gene_length=5, generations=5, mutation_rate=0.1, bounds=(-10,10)):
    pop = initialize_population(pop_size, gene_length, bounds)
    best_vals = []
    for g in range(1, generations+1):
        fitness = np.array([fitness_function(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_vals.append(fitness[best_idx])
        print(f"Generation {g} | Best fitness: {fitness[best_idx]:.6f}")
        parents = selection(pop, fitness)
        offspring = crossover(parents, pop_size)
        offspring = mutation(offspring, mutation_rate, bounds)
        pop = gene_expression(offspring)
    print("Best fitness value:", min(best_vals))

if __name__ == "__main__":
    gea(fitness_function, pop_size=10, gene_length=5, generations=5, mutation_rate=0.2, bounds=(-10,10))
