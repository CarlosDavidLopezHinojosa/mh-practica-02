import numpy as np


class gaussian:
    """
    Mutación gaussiana.
    Esta clase implementa la mutación gaussiana, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    sumando un valor aleatorio de una distribución normal con media 0 y desviación
    estándar `sigma`.
    Args:
        mutation_rate (float): Probabilidad de mutación por gen. Por defecto, 0.1.
        sigma (float): Desviación estándar de la mutación. Por defecto, 0.1.
    """
    def __init__(self, mutation_rate=0.1, sigma=0.1):
        self.mutation_rate = mutation_rate
        self.sigma = sigma
    def __call__(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutación gaussiana utilizando numpy.
        Args:
            individual (np.ndarray): Individuo a mutar.
            mutation_rate (float): Probabilidad de mutación por gen.
            sigma (float): Desviación estándar de la mutación.
        Returns:
            np.ndarray: Individuo mutado.
        """
        mutation_mask = np.random.random(individual.shape) < self.mutation_rate
        individual[mutation_mask] += np.random.normal(0, self.sigma, np.count_nonzero(mutation_mask))
        return individual

def mutations():
    return {
        "Mutación Gaussiana": gaussian,
    }