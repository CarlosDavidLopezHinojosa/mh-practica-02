import numpy as np


class gaussian:
    """
    Mutación gaussiana.
    Esta clase implementa la mutación gaussiana, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    sumando un valor aleatorio de una distribución normal con media 0 y desviación
    estándar `sigma`.
    Args:
        sigma (float): Desviación estándar de la mutación. Por defecto, 0.1.
    """
    def __init__(self, sigma=0.1):
        self.sigma = sigma
    def __call__(self, individual: np.ndarray, mutation_rate) -> np.ndarray:
        """
        Mutación gaussiana utilizando numpy.
        Args:
            individual (np.ndarray): Individuo a mutar.
            mutation_rate (float): Probabilidad de mutación por gen.
            sigma (float): Desviación estándar de la mutación.
        Returns:
            np.ndarray: Individuo mutado.
        """
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        individual[mutation_mask] += np.random.normal(0, self.sigma, np.count_nonzero(mutation_mask))
        return individual
    

class uniforme:
    """
    Mutación uniforme.
    Esta clase implementa la mutación uniforme, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    reemplazando el gen por un valor aleatorio entre 0 y 1.
    Args:
        mutation_rate (float): Probabilidad de mutación por gen. Por defecto, 0.1.
    """
    def __call__(self, individual: np.ndarray, mutation_rate) -> np.ndarray:
        """
        Mutación uniforme utilizando numpy.
        Args:
            individual (np.ndarray): Individuo a mutar.
            mutation_rate (float): Probabilidad de mutación por gen.
        Returns:
            np.ndarray: Individuo mutado.
        """
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        individual[mutation_mask] = np.random.uniform(0, 1, np.count_nonzero(mutation_mask))
        return individual
    

class no_uniforme:
    """
    Mutación no uniforme.
    Esta clase implementa la mutación no uniforme, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    reemplazando el gen por un valor aleatorio entre 0 y 1, pero con una distribución
    no uniforme.
    Args:
        mutation_rate (float): Probabilidad de mutación por gen. Por defecto, 0.1.
    """
    def __call__(self, individual: np.ndarray, mutation_rate) -> np.ndarray:
        """
        Mutación no uniforme utilizando numpy.
        Args:
            individual (np.ndarray): Individuo a mutar.
            mutation_rate (float): Probabilidad de mutación por gen.
        Returns:
            np.ndarray: Individuo mutado.
        """
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        individual[mutation_mask] = np.random.normal(0.5, 0.2, np.count_nonzero(mutation_mask))
        return individual
    

class polinomica:
    """
    Mutación polinómica.
    Esta clase implementa la mutación polinómica, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    utilizando una distribución polinómica.
    Args:
        mutation_rate (float): Probabilidad de mutación por gen. Por defecto, 0.1.
    """
    def __call__(self, individual: np.ndarray, mutation_rate) -> np.ndarray:
        """
        Mutación polinómica utilizando numpy.
        Args:
            individual (np.ndarray): Individuo a mutar.
            mutation_rate (float): Probabilidad de mutación por gen.
        Returns:
            np.ndarray: Individuo mutado.
        """
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        individual[mutation_mask] = np.random.pareto(1.5, np.count_nonzero(mutation_mask))
        return individual


def mutations():
    return {
        "Mutación Gaussiana": gaussian,
        "Mutación Uniforme": uniforme,
        "Mutación No Uniforme": no_uniforme,
        "Mutación Polinómica": polinomica,
    }