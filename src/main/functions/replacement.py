import numpy as np

class total:
    def __init__(self):
        pass
    def __call__(self, population, new_population):
        """
        Reemplazo total de la población.
        Args:
            population (np.ndarray): Población actual (matriz de individuos).
            new_population (np.ndarray): Nueva población generada (matriz de individuos).
        """
        np.copyto(population, new_population)

class worse:
    def __init__(self, fitness):
        self.fitness = fitness
        pass
    def __call__(self, population, new_population):
        """
        Reemplazo al peor de la población.
        Args:
            population (np.ndarray): Población actual (matriz de individuos).
            new_population (np.ndarray): Nueva población generada (matriz de individuos).
        """
        for new_individual in new_population:
            worst_index = np.argmax([self.fitness(ind) for ind in population])
            population[worst_index] = new_individual

class restricted_tournament:
    """
    Torneo restringido.
    Args:
        n (int): número de individuos en el torneo.
    """
    def __init__(self, n):
        self.n = n
    
    def __call__(self, population, new_population):
        """
        Realiza un torneo restringido escogiendo n candidatos aleatorios.
        Args:
            population (np.ndarray): Población actual (matriz de individuos).
            new_population (np.ndarray): Nueva población generada (matriz de individuos).
        """
        for new_individual in new_population:
            candidates_indices = np.random.choice(len(population), self.n, replace=False)
            candidates = population[candidates_indices]
            worst_index = np.argmax([np.sum(np.abs(candidates[i] - new_individual)) for i in range(len(candidates))])
            population[candidates_indices[worst_index]] = new_individual

class worse_between_similar:
    """
    Peor entre semejantes
    Args:
        n (int): número de individuos más parecidos entre los que reemplazar.
    """
    def __init__(self, n):
        self.n = n
    
    def __call__(self, population, new_population):
        """
        Realiza un reemplazo del peor entre semejantes.
        Args:
            population (np.ndarray): Población actual (matriz de individuos).
            new_population (np.ndarray): Nueva población generada (matriz de individuos).
        """
        for new_individual in new_population:
            distances = np.zeros(len(population)) 
            for i in range(len(population)):
                distances[i] = np.sum(np.abs(population[i] - new_individual))

            num_similar = min(self.n, len(population))
            similar_indices = np.argsort(distances)[:num_similar]
            similar_individuals = population[similar_indices]
            worst_index = np.argmax([np.sum(np.abs(similar_individuals[i] - new_individual)) for i in range(len(similar_individuals))])
            population[similar_indices[worst_index]] = new_individual


class elitism:

    def __init__(self, fitness):
        self.fitness = fitness
        pass

    def __call__(self, population: np.array, new_population: np.array):

        """
        Reemplazo elitista, mantiene los mejores individuos de la población actual.
        Args:
            population (np.ndarray): Población actual (matriz de individuos).
            new_population (np.ndarray): Nueva población generada (matriz de individuos).
        """
        # Suponiendo que la función de fitness es tal que un valor menor es mejor
        combined_population = np.vstack((population, new_population))
        fitness_values = np.array([self.fitness(ind) for ind in combined_population])
        best_indices = np.argsort(fitness_values)[:len(population)]
        population[:] = combined_population[best_indices]
        


def replacements():
    return {
        "Reemplazo Generacional Completo": total,
        "Reemplazar al peor de la población": worse,
        "Torneo restringido": restricted_tournament,
        "Peor entre semejantes": worse_between_similar,
        "Elitismo": elitism 
    }