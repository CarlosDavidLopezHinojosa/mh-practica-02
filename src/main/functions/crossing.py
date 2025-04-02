import numpy as np

def arithmetic(parent1, parent2):
    """
    Cruce aritmético entre dos padres utilizando numpy.
    Args:
        parent1 (np.ndarray): Primer padre.
        parent2 (np.ndarray): Segundo padre.
    Returns:
        np.ndarray: Hijo generado.
    """
    alpha = np.random.random()
    return alpha * parent1 + (1 - alpha) * parent2

def crossings():
    return {
        "Cruce Aritmético": arithmetic,
    }