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
    

def intercambio(individual: np.ndarray) -> np.ndarray:
    """
    Mutación basada en intercambios.
    Selecciona dos genes aleatorios del individuo y los intercambia.
    Args:
        individual (np.ndarray): Individuo a mutar.
    Returns:
        np.ndarray: Individuo mutado.
    """
    # Seleccionar dos índices aleatorios diferentes
    idx1, idx2 = np.random.choice(len(individual), size=2, replace=False)
    
    # Intercambiar los valores de los genes seleccionados
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    
    return individual


def insercion(individual: np.ndarray) -> np.ndarray:
    """
    Mutación basada en inserción.
    Selecciona un número n de genes consecutivos (n < 15% del tamaño del individuo),
    y los mueve a otra posición aleatoria manteniendo su orden y secuencia.
    Args:
        individual (np.ndarray): Individuo a mutar.
    Returns:
        np.ndarray: Individuo mutado.
    """
    size = len(individual)
    max_n = max(1, int(0.15 * size))  # Asegurarse de que n sea al menos 1
    n = np.random.randint(1, max_n + 1)  # Seleccionar n aleatorio entre 1 y max_n
    
    # Seleccionar el inicio del segmento de n genes
    start_idx = np.random.randint(0, size - n + 1)
    segment = individual[start_idx:start_idx + n]
    
    # Eliminar el segmento del individuo
    individual = np.delete(individual, slice(start_idx, start_idx + n))
    
    # Seleccionar una nueva posición aleatoria para insertar el segmento
    insert_idx = np.random.randint(0, len(individual) + 1)
    
    # Insertar el segmento en la nueva posición
    individual = np.insert(individual, insert_idx, segment)
    
    return individual


def inversion(individual: np.ndarray) -> np.ndarray:
    """
    Mutación basada en inversión.
    Selecciona un número n de genes consecutivos (n < 15% del tamaño del individuo),
    y los invierte manteniendo su posición original.
    Args:
        individual (np.ndarray): Individuo a mutar.
    Returns:
        np.ndarray: Individuo mutado.
    """
    size = len(individual)
    max_n = max(1, int(0.15 * size))  # Asegurarse de que n sea al menos 1
    n = np.random.randint(1, max_n + 1)  # Seleccionar n aleatorio entre 1 y max_n
    
    # Seleccionar el inicio del segmento de n genes
    start_idx = np.random.randint(0, size - n + 1)
    
    # Invertir el segmento
    individual[start_idx:start_idx + n] = np.flip(individual[start_idx:start_idx + n])
    
    return individual


def mutations():
    return {
        "Mutación Gaussiana": gaussian,
        "Mutación por Inserción": insercion,
    }