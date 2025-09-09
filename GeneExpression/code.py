import numpy as np

NUM_BEAMS = 5
GENES_PER_BEAM = 2
GENE_LENGTH = NUM_BEAMS * GENES_PER_BEAM

POP_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7

material_cost = [10, 6, 4]
material_strength = [20, 10, 5]

REQUIRED_STRENGTH = 100
PENALTY = 1e6

def init_population():
    population = []
    for _ in range(POP_SIZE):
        individual = []
        for _ in range(NUM_BEAMS):
            material = np.random.randint(0, 3)
            thickness = np.random.randint(1, 11)
            individual.extend([material, thickness])
        population.append(individual)
    return np.array(population)

def fitness_function(individual):
    total_cost = 0
    total_strength = 0
    for i in range(NUM_BEAMS):
        material = int(individual[2*i])
        thickness = int(individual[2*i + 1])
        cost = material_cost[material] * thickness
        strength = material_strength[material] * thickness
        total_cost += cost
        total_strength += strength
    if total_strength < REQUIRED_STRENGTH:
        return -PENALTY
    else:
        return total_strength / total_cost

def selection(population, fitnesses, k=3):
    selected = []
    for _ in range(POP_SIZE):
        aspirants_idx = np.random.choice(range(POP_SIZE), k)
        aspirants_fitness = fitnesses[aspirants_idx]
        winner_idx = aspirants_idx[np.argmax(aspirants_fitness)]
        selected.append(population[winner_idx])
    return np.array(selected)

def crossover(parent1, parent2):
    if np.random.rand() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    point = np.random.randint(1, GENE_LENGTH - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutate(individual):
    for i in range(0, GENE_LENGTH, 2):
        if np.random.rand() < MUTATION_RATE:
            individual[i] = np.random.randint(0, 3)
        if np.random.rand() < MUTATION_RATE:
            individual[i+1] = np.random.randint(1, 11)
    return individual

def run_gea():
    population = init_population()
    best_solution = None
    best_fitness = -np.inf

    for gen in range(NUM_GENERATIONS):
        fitnesses = np.array([fitness_function(ind) for ind in population])
        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > best_fitness:
            best_fitness = fitnesses[max_idx]
            best_solution = population[max_idx].copy()

        selected = selection(population, fitnesses)
        next_population = []
        for i in range(0, POP_SIZE, 2):
            p1 = selected[i]
            p2 = selected[i+1 if i+1 < POP_SIZE else 0]
            c1, c2 = crossover(p1, p2)
            next_population.append(mutate(c1))
            next_population.append(mutate(c2))
        population = np.array(next_population)

        if gen % 20 == 0 or gen == NUM_GENERATIONS - 1:
            print(f"Gen {gen}: Best fitness (strength/cost): {best_fitness:.4f}")

    print("\nBest bridge design (material, thickness per beam):")
    total_cost = 0
    total_strength = 0
    for i in range(NUM_BEAMS):
        material = int(best_solution[2*i])
        thickness = int(best_solution[2*i + 1])
        cost = material_cost[material] * thickness
        strength = material_strength[material] * thickness
        total_cost += cost
        total_strength += strength
        material_name = ["Steel", "Aluminum", "Concrete"][material]
        print(f" Beam {i+1}: Material={material_name}, Thickness={thickness}, Cost={cost}, Strength={strength}")

    print(f"Total cost: {total_cost}")
    print(f"Total strength: {total_strength} (Required: {REQUIRED_STRENGTH})")

if __name__ == "__main__":
    run_gea()
