import numpy as np
class tournament:
    """
    Selección por torneo binario.
    Esta clase implementa la selección por torneo binario, donde se seleccionan
    `k` individuos aleatorios de la población y se elige el mejor de ellos.
    El mejor individuo se selecciona utilizando una función de fitness proporcionada.
    Args:
        k (int): Número de individuos a seleccionar para el torneo. Por defecto, 2.
    """
    def __init__(self,k=2, fitness: callable):
        self.k = k
        self.fitness = fitness

    def select(self, population: np.array) -> np.array:
        """
        Selección por torneo binario.
        Args:
            population (np.ndarray): Población actual (matriz de individuos).
        Returns:
            np.ndarray: Individuo seleccionado.
        """
        indices = np.random.choice(len(population), self.k, replace=False)
        selected = population[indices]
        return min(selected, key=self.fitness)

    def __call__(self, population: np.array) -> np.array:
        """
        Llama a la función de selección.
        Args:
            population (np.ndarray): Población actual (matriz de individuos).
        Returns:
            np.ndarray: Individuo seleccionado.
        """
        p1 = self.select(population)
        p2 = self.select(population)
        while p2 is p1:
            p2 = self.select(population)

        return p1, p2

class random:
    """
    Selección aleatoria.
    Esta clase implementa la selección aleatoria, donde se seleccionan `n`individuos
    de manera aleatoria sin ninguna ponderación.
    Args:
        n (int): Número de individuos a seleccionar
    """
    def __init__(self, n=1):
        self.n = n
    
    def select(self, population: np.array) -> np.array:
        """
        Selección aleatoria.
        Args:
            population (np.ndarray): población actual (matriz de individuos).
        Returns:
            np.ndarray: Individuos seleccionados.
        """
        indices = np.random.choice(len(population), self.n, replace=False)
        return population[indices]
    
    def __call__(self, population: np.array) -> np.array:
        """
        Selección aleatoria utilizando numpy.
        Args:
            population (np.ndarray): población actual (matriz de individuos).
        Returns:
            np.ndarray: Individuos seleccionados.
        """
        p1 = self.select(population)
        p2 = self.select(population)
        while p2 is p1:
            p2 = self.select(population)
        return p1, p2

class roulette:
    """
    Selección por ruleta.
    Esta clase implementa la selección por ruleta, donde se distribuye una probabilidad total
    del 100% entre todos los individuos de la población en proporción a su resultado en la
    función de fitness y se seleccionan `n` individuos aleatoriamente con la probabilidad distribuida
    como ponderación.
    Args:
        n (int): Número de individuos a seleccionar
    """
    def __init__(self, n=1):
        self.n = n

    def __call__(population: np.array, fitness: callable, self) -> np.array:
        """
        Selección por ruleta utilizando numpy.
        Args:
            population (np.ndarray): población actual (matriz de individuos).
            fitness( callable): función de fitness.
        Returns:
            np.ndarray: Individuos seleccionados.
        """
        sum = 0
        fitness_array = np.empty(len(population))
        p = np.empty(len(population))
        
        for i in range(len(population)):
            fitness_array[i] = fitness(population[i])   # Guardamos el fitness de la población actual
        sum = np.sum(fitness_array)     # Obtenemos el sumatorio del fitness de la población actual
        for i in range(len(population)):
            p[i] = sum - fitness_array[i]   # Obtenemos el valor sum - fitness para cada individuo, haciendo que a mejor individuo, mejor p[i]
        sum = np.sum(p)     # Obtenemos el sumatorio de todos los sum - fitness de la población
        for i in range(len(population)):
            p[i] = p[i] / sum       # Distribuimos la probabilidad según sum - fitness
        
        return np.random.choice(len(population), size=self.n, replace=False, p=p)
    
class emparejamiento_variado_inverso:
    """
    Selección por emparejamiento variado inverso.
    Esta clase implementa la selección por emparejamiento variado inverso, que selecciona
    de manera aleatoria un primer padre de la población, y luego de otro subconjunto de padres
    seleccionado aleatoriamente selecciona al más distante al primero seleccionado, y devuelve
    a los dos.
    Args:
        k (int): número de padres en el subconjunto seleccionado.
    """
    def __init__(self, k = 2):
        self.k = k

    def __call__(self, population: np.array, fitness) -> np.array:
        """
        Selección por emparejamiento variado inverso.
        Args:
            population (np.ndarray): población actual (matriz de individuos).
        Returns:
            np.ndarray: los dos individuos seleccionados.
        """
        distance = np.zeros(len(population))
        first = random(population, n=1)     # Seleccionamos el primer padre de manera aleatoria
        seconds = random(population, n=self.k)      # Seleccionamos al subconjunto de candidatos para padres
        for i in len(seconds):
            for j in len(seconds[i]):    # Obtenemos la distancia entre el primer padre y el resto de individuos
                distance[i] += abs(first[j] - seconds[i][j])
        second_index = np.argmax(distance)  
        second = seconds[second_index]  # Seleccionamos el padre de mayor distancia
        return np.array([first, second]) 

def selections():
    return {
        "Aleatorio": random,
        "Torneo Binario": tournament,
        "Ruleta": roulette,
        "Emparejamiento variado inverso": emparejamiento_variado_inverso
    }
