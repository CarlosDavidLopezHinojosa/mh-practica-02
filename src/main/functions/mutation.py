import numpy as np


class gaussian:
    """
    Mutación gaussiana.
    Esta clase implementa la mutación gaussiana, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    sumando un valor aleatorio de una distribución normal con media 0 y desviación
    estándar `sigma`.
    Args:
        sigma (float): Desviación estándar de la mutación. Por defecto, 0.1.
    """
    def __init__(self, sigma=0.1):
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
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        indices = np.where(mutation_mask)[0]
        if len(indices) > 0:
            individual[indices] += np.random.normal(0, self.sigma, np.count_nonzero(mutation_mask))
        return individual
    

class uniform:
    """
    Mutación uniforme.
    Esta clase implementa la mutación uniforme, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    reemplazando el gen por un valor aleatorio entre 0 y 1.
    Args:
        mutation_rate (float): Probabilidad de mutación por gen. Por defecto, 0.1.
    """
    def __call__(self, individual: np.ndarray, mutation_rate) -> np.ndarray:
        """
        Mutación uniforme utilizando numpy.
        Args:
            individual (np.ndarray): Individuo a mutar.
            mutation_rate (float): Probabilidad de mutación por gen.
        Returns:
            np.ndarray: Individuo mutado.
        """
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        indices = np.where(mutation_mask)[0]
        if len(indices) > 0:
            individual[indices] += np.random.uniform(-1, 1, np.count_nonzero(mutation_mask))
        return individual
    

class non_uniform:
    """
    Mutación no uniforme.
    Esta clase implementa la mutación no uniforme, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    ajustando los valores de los genes hacia los límites mínimo o máximo de manera
    no uniforme.
    Args:
        timer_max (int): Número máximo de iteraciones (generaciones).
        speed (float): Velocidad de convergencia de la mutación.
    """
    def __init__(self, timer_max, speed):
        self.minimum = None
        self.maximum = None
        self.timer_max = timer_max
        self.speed = speed
        self.timer = 0

    def reducer(self, r):
        """
        Calcula el factor de reducción basado en el progreso de las generaciones.
        """
        return 1 - r ** ((1 - self.timer / self.timer_max) ** self.speed)

    def low_delta(self, individual, indices):
        """
        Calcula el delta para mover los genes hacia el límite inferior.
        """
        if self.minimum is None:
            raise ValueError("El límite inferior (minimum) no está definido.")
        return individual[indices] - ((individual[indices] - self.minimum[indices]) * self.reducer(np.random.random()))

    def high_delta(self, individual, indices):
        """
        Calcula el delta para mover los genes hacia el límite superior.
        """
        if self.maximum is None:
            raise ValueError("El límite superior (maximum) no está definido.")
        return individual[indices] + ((self.maximum[indices] - individual[indices]) * self.reducer(np.random.random()))

    def delta(self, individual, indices):
        """
        Decide si aplicar `low_delta` o `high_delta` para los genes seleccionados.
        """
        return np.where(
            np.random.random(len(indices)) < 0.5,
            self.low_delta(individual, indices),
            self.high_delta(individual, indices)
        )

    def __call__(self, individual: np.ndarray, mutation_rate) -> np.ndarray:
        """
        Aplica la mutación no uniforme al individuo.
        Args:
            individual (np.ndarray): Individuo a mutar.
            mutation_rate (float): Probabilidad de mutación por gen.
        Returns:
            np.ndarray: Individuo mutado.
        """
        # Inicializar límites si no están definidos
        
        self.minimum = np.copy(individual) if self.minimum is None else np.minimum(self.minimum, individual)
        self.maximum = np.copy(individual) if self.maximum is None else np.maximum(self.maximum, individual)

        # Generar máscara de mutación
        mutation_mask = np.random.random(individual.shape) < mutation_rate
        indices = np.where(mutation_mask)[0]
        if len(indices) > 0:
            individual[indices] = self.delta(individual, indices)
        
        self.timer = min(self.timer + 1, self.timer_max) 
        return individual
        
    

class polynomial:
    """
    Mutación polinómica.
    Esta clase implementa la mutación polinómica, donde cada gen del individuo
    tiene una probabilidad `mutation_rate` de ser mutado. La mutación se realiza
    utilizando una distribución polinómica.
    Args:
        mutation_rate (float): Probabilidad de mutación por gen. Por defecto, 0.1.
    """
    def __init__(self, eta=20):
        self.eta = eta
        self.minimum = None
        self.maximum = None

    def low_delta(self, p):
        return (2 * p) ** (1.0 / (self.eta + 1)) - 1
    
    def high_delta(self, p):
        return 1 - (2 * (1 - p)) ** (1.0 / (self.eta + 1))
    
    def delta(self, p):
        if p < 0.5:
            return self.low_delta(p)
        else:
            return self.high_delta(p)

    def __call__(self, individual: np.ndarray, mutation_rate) -> np.ndarray:
        """
        Mutación polinómica utilizando numpy.
        Args:
            individual (np.ndarray): Individuo a mutar.
            mutation_rate (float): Probabilidad de mutación por gen.
        Returns:
            np.ndarray: Individuo mutado.
        """
        self.minimum = np.copy(individual) if self.minimum is None else np.minimum(self.minimum, individual)
        self.maximum = np.copy(individual) if self.maximum is None else np.maximum(self.maximum, individual)

        # Generar máscara de mutación
        mutation_mask = np.random.random(individual.shape) < mutation_rate

        # Generar valores aleatorios para calcular deltas
        ps = np.random.random(np.count_nonzero(mutation_mask))
        deltas = np.array([self.delta(p) for p in ps])

        # Aplicar deltas solo a los genes seleccionados por la máscara
        individual[mutation_mask] += deltas * (self.maximum[mutation_mask] - self.minimum[mutation_mask])
        return individual

        


def mutations():
    return {
        "Mutación Gaussiana": gaussian,
        "Mutación Uniforme": uniform,
        "Mutación No Uniforme": non_uniform,
        "Mutación Polinómica": polynomial,
    }