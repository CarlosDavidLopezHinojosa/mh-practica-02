# Selección: Torneo binario
import numpy as np

def tournament(population, fitness, k=2):
    """
    Selección por torneo binario utilizando numpy.
    Args:
        population (np.ndarray): Población actual (matriz de individuos).
        fitness (callable): Función de fitness.
        k (int): Número de individuos en el torneo.
    Returns:
        np.ndarray: Individuo seleccionado.
    """
    indices = np.random.choice(len(population), k, replace=False)  # Selecciona índices aleatorios
    selected = population[indices]  # Extrae los individuos correspondientes
    fitness_values = np.apply_along_axis(fitness, 1, selected)  # Calcula fitness en un solo paso
    return selected[np.argmin(fitness_values)]  # Devu

def selections():
    return [tournament]