import numpy as np

def total(population, new_population, fitness):
    """
    Reemplazo total de la población.
    Args:
        population (np.ndarray): Población actual (matriz de individuos).
        new_population (np.ndarray): Nueva población generada (matriz de individuos).
    """
    np.copyto(population, new_population)

def replacements():
    return {
        "Reemplazo Generacional Completo": total,
    }