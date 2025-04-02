import numpy as np
class tournament:
    """
    Selección por torneo binario.
    Esta clase implementa la selección por torneo binario, donde se seleccionan
    `k` individuos aleatorios de la población y se elige el mejor de ellos.
    El mejor individuo se selecciona utilizando una función de fitness proporcionada.
    Args:
        k (int): Número de individuos a seleccionar para el torneo. Por defecto, 2.
    """
    def __init__(self,k=2):
        self.k = k

    def __call__(self, population: np.array, fitness: callable) -> np.array:
        """
        Selección por torneo binario utilizando numpy.
        Args:
            population (np.ndarray): Población actual (matriz de individuos).
            fitness (callable): Función de fitness.
        Returns:
            np.ndarray: Individuo seleccionado.
        """
        indices = np.random.choice(len(population), self.k, replace=False)
        selected = population[indices]
        return min(selected, key=fitness)
        

def selections():
    return {
        "Torneo Binario": tournament,
    }
