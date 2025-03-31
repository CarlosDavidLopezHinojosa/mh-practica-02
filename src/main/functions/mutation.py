import numpy as np

def gaussian(individual, mutation_rate=0.1, sigma=0.1):
    """
    Mutación gaussiana utilizando numpy.
    Args:
        individual (np.ndarray): Individuo a mutar.
        mutation_rate (float): Probabilidad de mutación por gen.
        sigma (float): Desviación estándar de la mutación.
    Returns:
        np.ndarray: Individuo mutado.
    """
    mutation_mask = np.random.random(individual.shape) < mutation_rate  # Genes a mutar
    individual[mutation_mask] += np.random.normal(0, sigma, np.count_nonzero(mutation_mask))
    return individual

def mutations():
    return [gaussian]