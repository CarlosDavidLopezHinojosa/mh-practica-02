import numpy as np

class tournament:
    def __init__(self, fitness: callable, k=2):
        self.k = k
        self.fitness = fitness

    def select(self, population: np.ndarray) -> np.ndarray:
        indices = np.random.choice(len(population), self.k, replace=False)
        selected = population[indices]
        return min(selected, key=self.fitness)

    def __call__(self, population: np.ndarray) -> tuple:
        p1 = self.select(population)
        p2 = self.select(population)
        while np.array_equal(p1, p2):
            p2 = self.select(population)
        return p1, p2

class random:
    def __init__(self, n=1):
        self.n = n
    
    def select(self, population: np.ndarray) -> np.ndarray:
        indices = np.random.choice(len(population), self.n, replace=False)
        return population[indices]

    def __call__(self, population: np.ndarray) -> tuple:
        selected = self.select(population)
        if self.n == 1:
            p1 = selected[0]
            while True:
                p2 = self.select(population)[0]
                if not np.array_equal(p1, p2):
                    break
            return p1, p2
        elif self.n >= 2:
            return selected[0], selected[1]

class roulette:
    def __init__(self, fitness: callable, n=1):
        self.n = n
        self.fitness = fitness

    def __call__(self, population: np.ndarray) -> tuple:
        fitness_values = np.array([self.fitness(ind) for ind in population])
        inverse_fitness = np.max(fitness_values) - fitness_values + 1e-6
        probabilities = inverse_fitness / np.sum(inverse_fitness)

        indices = np.random.choice(len(population), size=self.n, replace=False, p=probabilities)
        if self.n == 1:
            p1 = population[indices[0]]
            while True:
                idx = np.random.choice(len(population), p=probabilities)
                p2 = population[idx]
                if not np.array_equal(p1, p2):
                    break
            return p1, p2
        else:
            return population[indices[0]], population[indices[1]]

class inverse_diversity_matching:
    def __init__(self, k=2):
        self.k = k

    def __call__(self, population: np.ndarray) -> tuple:
        first_idx = np.random.choice(len(population))
        first = population[first_idx]

        candidate_indices = np.random.choice(len(population), self.k, replace=False)
        candidates = population[candidate_indices]

        distances = np.array([
            np.sum(np.abs(first - candidate)) for candidate in candidates
        ])
        second = candidates[np.argmax(distances)]
        return first, second

def selections():
    return {
        "Aleatorio": random,
        "Torneo Binario": tournament,
        "Ruleta": roulette,
        "Emparejamiento variado inverso": inverse_diversity_matching
    }
