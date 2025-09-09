import random
import math

# --- CONFIGURATION ---
NUM_CITIES = 10
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
MUTATION_RATE = 0.015
TOURNAMENT_SIZE = 5
ELITISM = True


# --- CITY CLASS ---
class City:
    def __init__(self):
        self.x = random.randint(0, 100)
        self.y = random.randint(0, 100)

    def distance_to(self, city):
        dx = self.x - city.x
        dy = self.y - city.y
        return math.sqrt(dx * dx + dy * dy)

    def __repr__(self):
        return f"({self.x},{self.y})"


# --- TOUR CLASS (a possible solution) ---
class Tour:
    def __init__(self, cities):
        self.cities = cities[:]
        random.shuffle(self.cities)
        self.fitness = 0
        self.distance = 0

    def copy(self):
        t = Tour(self.cities[:])
        t.fitness = self.fitness
        t.distance = self.distance
        return t

    def get_distance(self):
        if self.distance == 0:
            total = 0
            for i in range(len(self.cities)):
                from_city = self.cities[i]
                to_city = self.cities[(i + 1) % len(self.cities)]
                total += from_city.distance_to(to_city)
            self.distance = total
        return self.distance

    def get_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / self.get_distance()
        return self.fitness

    def mutate(self):
        for i in range(len(self.cities)):
            if random.random() < MUTATION_RATE:
                j = random.randint(0, len(self.cities) - 1)
                self.cities[i], self.cities[j] = self.cities[j], self.cities[i]
        self.fitness = 0
        self.distance = 0

    def __repr__(self):
        return str(self.cities)


# --- POPULATION CLASS ---
class Population:
    def __init__(self, size, all_cities):
        self.tours = []
        if all_cities:
            for _ in range(size):
                self.tours.append(Tour(all_cities))
        else:
            self.tours = [None] * size

    def get_fittest(self):
        return max(self.tours, key=lambda t: t.get_fitness())


# --- GA OPERATIONS ---
def crossover(parent1, parent2):
    size = len(parent1.cities)
    child = [None] * size

    start, end = sorted([random.randint(0, size - 1), random.randint(0, size - 1)])

    for i in range(start, end):
        child[i] = parent1.cities[i]

    current = 0
    for i in range(size):
        city = parent2.cities[i]
        if city not in child:
            while child[current] is not None:
                current += 1
            child[current] = city

    t = Tour(child)
    return t


def tournament_selection(pop):
    tournament = Population(TOURNAMENT_SIZE, [])
    for i in range(TOURNAMENT_SIZE):
        rand_idx = random.randint(0, len(pop.tours) - 1)
        tournament.tours[i] = pop.tours[rand_idx]
    return tournament.get_fittest()


def evolve_population(pop):
    new_pop = Population(len(pop.tours), [])
    offset = 0

    if ELITISM:
        new_pop.tours[0] = pop.get_fittest()
        offset = 1

    for i in range(offset, len(pop.tours)):
        parent1 = tournament_selection(pop)
        parent2 = tournament_selection(pop)
        child = crossover(parent1, parent2)
        child.mutate()
        new_pop.tours[i] = child

    return new_pop


# --- MAIN ---
if __name__ == "__main__":
    # Create cities
    cities = [City() for _ in range(NUM_CITIES)]

    # Initialize population
    pop = Population(POPULATION_SIZE, cities)
    print("Initial distance:", pop.get_fittest().get_distance())

    # Evolve
    for i in range(NUM_GENERATIONS):
        pop = evolve_population(pop)
        if i % 50 == 0 or i == NUM_GENERATIONS - 1:
            print(f"Generation {i}: Best distance = {pop.get_fittest().get_distance():.2f}")

    # Final result
    best = pop.get_fittest()
    print("\nFinal best distance:", best.get_distance())
    print("Best tour:", best)
