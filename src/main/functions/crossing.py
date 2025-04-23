import numpy as np

class crosser:
    """
    Clase base para la selección de individuos.
    Esta clase es una interfaz para las diferentes estrategias de selección de individuos.
    """
    def __init__(self, fitness: callable, mode: bool = False):
        self.fitness = fitness
        self.mode = mode
        self.convengences = []
    
    def convergences(self):
        """
        Devuelve la lista de convergencias.
        Returns:
            list: Lista de convergencias.
        """
        return self.convengences

class arithmetic(crosser):
    
    def __init__(self, fitness, mode = False):
        super().__init__(fitness, mode)
    
    def __call__(self, parent1, parent2):
        """
        Cruce aritmético entre dos padres utilizando numpy.
        Args:
            parent1 (np.ndarray): Primer padre.
            parent2 (np.ndarray): Segundo padre.
        Returns:
            np.ndarray: Hijo generado.
        """
        alpha = np.random.random()
        ch = alpha * parent1 + (1 - alpha) * parent2

        if self.mode:
            self.convengences.append(float(self.fitness(ch)))
        return ch


class single_point(crosser):

    def __init__(self, fitness, mode = False):
        super().__init__(fitness, mode)
    
    def __call__(self, parent1, parent2):
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
        if self.mode:
            self.convengences.append(float(self.fitness(child)))
        return child


class uniform(crosser):
    def __init__(self, fitness, mode = False):
        super().__init__(fitness, mode)
    
    def __call__(self, parent1, parent2):
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
        if self.mode:
            self.convengences.append(float(self.fitness(child)))
        return child


class BLX(crosser):
    def __init__(self, fitness, mode = False):
        super().__init__(fitness, mode)
    
    def __call__(self, parent1, parent2):
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
        if self.mode:
            self.convengences.append(float(self.fitness(child)))
        return child


def crossings():
    return {
        "Cruce Aritmético": arithmetic,
        "Cruce de Un Punto": single_point,
        "Cruce Uniforme": uniform,
        "Cruce BLX": BLX,
    }