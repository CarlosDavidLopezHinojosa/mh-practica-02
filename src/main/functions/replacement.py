import numpy as np

class replacer:
    """
    Clase base para la selección de individuos.
    Esta clase es una interfaz para las diferentes estrategias de selección de individuos.
    """
    def __init__(self, fitness: callable, mode: bool = False):
        self.fitness = fitness
        self.mode = mode
        self.convengences = []
    
    def convergences(self):
        """
        Devuelve la lista de convergencias.
        Returns:
            list: Lista de convergencias.
        """
        return self.convengences

class total(replacer):
    def __init__(self, fitness, mode = False):
        super().__init__(fitness, mode)
    def __call__(self, population, new_population):
        """
        Reemplazo total de la población.
        Args:
            population (np.ndarray): Población actual (matriz de individuos).
            new_population (np.ndarray): Nueva población generada (matriz de individuos).
        """
        np.copyto(population, new_population)
        if self.mode:
            self.convengences.append(min([float(self.fitness(p)) for p in population]))

class worse(replacer):
    def __init__(self, fitness, mode = False):
        super().__init__(fitness, mode)
    
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

        if self.mode:
            self.convengences.append(min([float(self.fitness(p)) for p in population]))

class restricted_tournament(replacer):
    """
    Torneo restringido.
    Args:
        n (int): número de individuos en el torneo.
    """
    def __init__(self, n, fitness, mode = False):
        super().__init__(fitness, mode)
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
            
        if self.mode:
            self.convengences.append(min([float(self.fitness(p)) for p in population]))
class worse_between_similar(replacer):
    """
    Peor entre semejantes
    Args:
        n (int): número de individuos más parecidos entre los que reemplazar.
    """
    def __init__(self, n, fitness, mode = False):
        super().__init__(fitness, mode)
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

        if self.mode:
            self.convengences.append(min(float(self.fitness(p)) for p in population))


class elitism(replacer):

    def __init__(self, fitness, mode = False):
        super().__init__(fitness, mode)

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
        if self.mode:
            self.convengences.append(self.fitness(population[0]))
        


def replacements():
    return {
        "Reemplazo Generacional Completo": total,
        "Reemplazar al peor de la población": worse,
        "Torneo restringido": restricted_tournament,
        "Peor entre semejantes": worse_between_similar,
        "Elitismo": elitism 
    }