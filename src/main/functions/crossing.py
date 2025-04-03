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


def uniform(parent1, parent2):
    """
    Cruce uniforme entre dos padres utilizando numpy.
    Args:
        parent1 (np.ndarray): Primer padre.
        parent2 (np.ndarray): Segundo padre.
    Returns:
        np.ndarray: Hijo generado.
    """
    mask = np.random.randint(0, 2, size=parent1.shape).astype(bool)
    child = np.where(mask, parent1, parent2)
    return child


def BLX(parent1, parent2):
    """
    Cruce BLX entre dos padres utilizando numpy.
    Args:
        parent1 (np.ndarray): Primer padre.
        parent2 (np.ndarray): Segundo padre.
    Returns:
        np.ndarray: Hijo generado.
    """
    alpha = 0.5
    lower_bound = np.minimum(parent1, parent2) - alpha * np.abs(parent1 - parent2)
    upper_bound = np.maximum(parent1, parent2) + alpha * np.abs(parent1 - parent2)
    child = np.random.uniform(lower_bound, upper_bound)
    return child


def crossings():
    return {
        "Cruce Aritmético": arithmetic,
    }