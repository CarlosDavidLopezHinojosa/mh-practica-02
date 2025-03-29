import random

# Selección: Torneo binario
def selection_tournament(population, fitness, k=2):
    """
    Selección por torneo binario.
    Args:
        population (list): Población actual.
        fitness (callable): Función de fitness.
        k (int): Número de individuos en el torneo.
    Returns:
        list: Individuo seleccionado.
    """
    tournament = random.sample(list(population), k)
    return min(tournament, key=fitness)

# Cruce: Cruce aritmético
def crossing_arithmetic(parent1, parent2):
    """
    Cruce aritmético entre dos padres.
    Args:
        parent1 (list): Primer padre.
        parent2 (list): Segundo padre.
    Returns:
        list: Hijo generado.
    """
    alpha = random.random()
    return [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]

# Mutación: Mutación gaussiana
def mutation_gaussian(individual, mutation_rate=0.1, sigma=0.1):
    """
    Mutación gaussiana.
    Args:
        individual (list): Individuo a mutar.
        mutation_rate (float): Probabilidad de mutación por gen.
        sigma (float): Desviación estándar de la mutación.
    Returns:
        list: Individuo mutado.
    """
    return [
        gene + random.gauss(0, sigma) if random.random() < mutation_rate else gene
        for gene in individual
    ]

def get_fitness_key(individual):
    return individual[1]

# Reemplazo: Elitismo
def replacement_elitism(population, new_population):
    """
    Reemplazo con elitismo.
    Args:
        population (list): Población actual.
        new_population (list): Nueva población generada.
    """
    population.sort(key=get_fitness_key)
    new_population.sort(key=get_fitness_key)
    for i in range(len(new_population)):
        population[i] = new_population[i]
