import numpy as np

def elitism(population, new_population, fitness):
    """
    Reemplazo con elitismo utilizando numpy.
    Args:
        population (np.ndarray): Población actual (matriz de individuos).
        new_population (np.ndarray): Nueva población generada (matriz de individuos).
        fitness (callable): Función de fitness.
    """
    combined_population = np.concatenate((population, new_population), axis=0)
    fitness_values = np.apply_along_axis(fitness, 1, combined_population)

    # Seleccionar los individuos con mejor fitness
    best_indices = np.argsort(fitness_values)[:len(population)]
    np.copyto(population, combined_population[best_indices])

    # Liberar memoria explícitamente
    del combined_population
    del fitness_values

def replacements():
    return [elitism]