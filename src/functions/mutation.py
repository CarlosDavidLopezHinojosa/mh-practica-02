import numpy as np
from tools.utils import instant, memory, memstart, memstop


class mutator:
    """
    Clase base para la selección de individuos.
    Esta clase es una interfaz para las diferentes estrategias de selección de individuos.
    """
    def __init__(self, fitness: callable, mode: bool = False):
        self.fitness = fitness
        self.mode = mode
        self.measures = {'time': [], 'memory': [], 'convergences': []}
        

        
class gaussian(mutator):
    """
    Mutación gaussiana.
    Esta clase implementa la mutación gaussiana, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    sumando un valor aleatorio de una distribución normal con media 0 y desviación
    estándar `sigma`.
    Args:
        sigma (float): Desviación estándar de la mutación. Por defecto, 0.1.
    """
    def __init__(self, sigma, fitness, mode = False):
        super().__init__(fitness, mode)
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
        if self.mode:
            memstart()
            start = instant()
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        individual[mutation_mask] += np.random.normal(0, self.sigma, np.count_nonzero(mutation_mask))

        if self.mode:
            self.measures['time'].append(instant() - start)
            self.measures['memory'].append(memory())
            memstop()
            self.measures['convergences'].append(float(self.fitness(individual)))
        return individual
    

class uniform(mutator):
    """
    Mutación uniforme.
    Esta clase implementa la mutación uniforme, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    reemplazando el gen por un valor aleatorio entre 0 y 1.
    Args:
        mutation_rate (float): Probabilidad de mutación por gen. Por defecto, 0.1.
    """
    def __init__(self, fitness, mode = False):
        super().__init__(fitness, mode)

    def __call__(self, individual: np.ndarray, mutation_rate) -> np.ndarray:
        """
        Mutación uniforme utilizando numpy.
        Args:
            individual (np.ndarray): Individuo a mutar.
            mutation_rate (float): Probabilidad de mutación por gen.
        Returns:
            np.ndarray: Individuo mutado.
        """

        if self.mode:
            memstart()
            start = instant()
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        individual[mutation_mask] = np.random.uniform(0, 1, np.count_nonzero(mutation_mask))
        if self.mode:
            self.measures['time'].append(instant() - start)
            self.measures['memory'].append(memory())
            memstop()
            self.measures['convergences'].append(float(self.fitness(individual)))
        return individual
    

class non_uniform(mutator):
    """
    Mutación no uniforme adaptada para coeficientes pequeños.
    Usa una distribución normal escalada y suaviza la decaimiento.
    """
    def __init__(self, max_gen, fitness, mode=False):
        super().__init__(fitness, mode)
        self.max_gen = max_gen
        self.current_gen = 0
        self.sigma = 0.01  # Controla magnitud máxima inicial
        self.decay_rate = 2  # Controla ritmo de decaimiento (2-5)

    def __call__(self, individual: np.ndarray, mutation_rate: float) -> np.ndarray:
        if self.mode:
            memstart()
            start = instant()

        mutation_mask = np.random.random(individual.shape) < mutation_rate
        t = self.current_gen / self.max_gen
        
        # Calcular factor de decaimiento no lineal
        decay_factor = (1 - t)**self.decay_rate
        
        # Generar mutaciones con distribución normal escalada
        deltas = np.random.normal(
            scale=self.sigma * decay_factor,
            size=individual.shape
        )
        
        direction = np.random.choice([-1, 1], size=individual.shape)
        deltas = deltas * direction
        # Aplicar mutaciones solo donde el mask es True
        individual[mutation_mask] += deltas[mutation_mask]
        
        # Suavizar el clipping para no perder información
        individual = np.clip(individual, -5.0, 5.0)  # Asumiendo rango [-1,1]
        
        self.current_gen += 1

        if self.mode:
            self.measures['time'].append(instant() - start)
            self.measures['memory'].append(memory())
            memstop()
            self.measures['convergences'].append(float(self.fitness(individual)))
        
        return individual
    

class polinomica(mutator):
    """
    Mutación polinómica.
    Esta clase implementa la mutación polinómica, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    utilizando una distribución polinómica.
    Args:
        mutation_rate (float): Probabilidad de mutación por gen. Por defecto, 0.1.
    """

    def __init__(self, fitness, mode = False):
        super().__init__(fitness, mode)

    def __call__(self, individual: np.ndarray, mutation_rate) -> np.ndarray:
        """
        Mutación polinómica utilizando numpy.
        Args:
            individual (np.ndarray): Individuo a mutar.
            mutation_rate (float): Probabilidad de mutación por gen.
        Returns:
            np.ndarray: Individuo mutado.
        """
        if self.mode:
            memstart()
            start = instant()
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        direction = np.random.choice([-1, 1], size=individual.shape)
        pareto = np.random.pareto(4.7, np.count_nonzero(mutation_mask))
        individual[mutation_mask] += pareto * direction[mutation_mask]
        if self.mode:
            self.measures['time'].append(instant() - start)
            self.measures['memory'].append(memory())
            memstop()

            self.measures["convergences"].append(float(self.fitness(individual)))
        return individual


def mutations():
    return {
        "Mutación Gaussiana": gaussian,
        "Mutación Uniforme": uniform,
        "Mutación No Uniforme": non_uniform,
        "Mutación Polinómica": polinomica,
    }