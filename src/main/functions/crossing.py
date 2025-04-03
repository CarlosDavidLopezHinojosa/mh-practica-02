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


def single_point(parent1, parent2):
    """
    Cruce de un punto entre dos padres utilizando numpy.
    Args:
        parent1 (np.ndarray): Primer padre.
        parent2 (np.ndarray): Segundo padre.
    Returns:
        np.ndarray: Hijo generado.
    """
    point = np.random.randint(1, len(parent1))
    child = np.concatenate((parent1[:point], parent2[point:]))
    return child


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
    alpha = np.random.random()
    lower_bound = np.minimum(parent1, parent2) - alpha * np.abs(parent1 - parent2)
    upper_bound = np.maximum(parent1, parent2) + alpha * np.abs(parent1 - parent2)
    child = np.random.uniform(lower_bound, upper_bound)
    return child


def crossings():
    return {
        "Cruce Aritmético": arithmetic,
        "Cruce de Un Punto": single_point,
        "Cruce Uniforme": uniform,
        "Cruce BLX": BLX,
    }