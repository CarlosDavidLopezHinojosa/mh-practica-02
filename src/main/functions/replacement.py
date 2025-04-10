import numpy as np

def total(population, new_population, fitness):
    """
    Reemplazo total de la población.
    Args:
        population (np.ndarray): Población actual (matriz de individuos).
        new_population (np.ndarray): Nueva población generada (matriz de individuos).
    """
    np.copyto(population, new_population)

def worse(population, new_population, fitness):
    """
    Reemplazo al peor de la población.
    Args:
        population (np.ndarray): Población actual (matriz de individuos).
        new_population (np.ndarray): Nueva población generada (matriz de individuos).
    """
    for new_individual in new_population:
        worst_index = np.argmax([fitness(ind) for ind in population])
        population[worst_index] = new_individual

class restricted_tournament:
    """
    Torneo restringido.
    Args:
        n (int): número de individuos en el torneo.
    """
    def __init__(self, n):
        self.n = n
    
    def __call__(self, population, new_population, fitness):
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
    
    def __call__(self, population, new_population, fitness):
        """
        Realiza un reemplazo del peor entre semejantes.
        Args:
            population (np.ndarray): Población actual (matriz de individuos).
            new_population (np.ndarray): Nueva población generada (matriz de individuos).
        """
        for new_individual in new_population:
            distances = np.zeros(self.n)
            for i in range(len(population)):
                distances[i] = np.sum(np.abs(population[i] - new_individual))
            similar_indices = np.argsort(distances)[:self.n]
            similar_individuals = population[similar_indices]
            worst_index = np.argmax([np.sum(np.abs(similar_individuals[i] - new_individual)) for i in range(len(similar_individuals))])
            population[similar_indices[worst_index]] = new_individual

def replacements():
    return {
        "Reemplazo Generacional Completo": total,
        "Reemplazar al peor de la población": worse,
        "Torneo restringido": restricted_tournament,
        "Peor entre semejantes": worse_between_similar
    }