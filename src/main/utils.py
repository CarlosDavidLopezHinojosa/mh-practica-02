import time
import tracemalloc as tm
import random as rnd

from functools import wraps
from math import exp, log, sqrt

def random_coeficients(n: int) -> list:
    """
    Genera un array de coeficientes aleatorios.
    
    Args:
        n (int): Número de coeficientes a generar.
    
    Returns:
        list: Array de tamaño `n` con valores aleatorios entre 0 y 1.
    """
    return [rnd.random() for _ in range(n)]

def initial_population(pop_size: int, n: int) -> list:
    """
    Genera una población inicial de individuos con coeficientes aleatorios.
    
    Args:
        pop_size (int): Tamaño de la población (número de individuos).
        n (int): Número de coeficientes por individuo.
    
    Returns:
        list: Matriz de tamaño `(pop_size, n)` con coeficientes aleatorios.
    """
    return [random_coeficients(n) for _ in range(pop_size)]

def create_islands(num_island, pop_size, num_coefs):
    """
    Genera una lista de poblaciones para el modelo de islas.
    
    Args:
        num_island (int): Número de islas.
        pop_size (int): Tamaño de la población por isla.
        num_coefs (int): Número de coeficientes por individuo.
    
    Returns:
        list: Lista de poblaciones para cada isla.
    """
    return [initial_population(pop_size, num_coefs) for _ in range(num_island)]

def get_data(lower: int = 0, upper: int = 15) -> list:
    """
    Obtiene un subconjunto de datos de prueba en el rango especificado.
    
    Args:
        lower (int, opcional): Índice inferior del subconjunto. Por defecto, 0.
        upper (int, opcional): Índice superior del subconjunto. Por defecto, 15.
    
    Returns:
        list: Matriz de datos en formato `[x, y]` con valores en el rango `[lower:upper]`.
    """
    return [
        [0, 3.490342957],
        [0.1, 3.649406057],
        [0.2, 3.850310157],
        [0.3, 4.110680257],
        [0.4, 4.444613357],
        [0.5, 4.864490457],
        [1, 8.945762957],
        [1.3, 14.907555257],
        [-0.1, 3.3508758574],
        [-1.6, -10.443986642],
        [-1.7, -10.134869742],
        [-0.83, -0.0700854481],
        [-0.82, 0.0372406176],
        [-1.98, -0.2501897336],
        [-1.99, 0.4626335969]
    ][lower:upper]

def f(x: float, coeficients: list) -> float:
    """
    Calcula el valor de una función basada en coeficientes polinomiales.
    
    La función tiene la forma:
        f(x) = exp(coeficients[0]) + coeficients[1]*x^1 + coeficients[2]*x^2 + ...
    
    Args:
        x (float): Valor de entrada.
        coeficients (list): Lista de coeficientes del modelo.
    
    Returns:
        float: Resultado de la función para el valor `x` dado.
    """
    return exp(coeficients[0]) + sum([coeficients[i] * x ** i for i in range(1, len(coeficients))])

def error(coeficients: list, data: list) -> float:
    """
    Calcula el error cuadrático total entre los valores estimados y los valores reales.
    
    Args:
        coeficients (list): Coeficientes del modelo.
        data (list): Matriz de datos en formato `[x, y]`, donde:
                         - `x` es el valor de entrada.
                         - `y` es el valor real esperado.
    
    Returns:
        float: Error cuadrático total.
    """
    return sum([(f(x, coeficients) - y) ** 2 for x, y in data])

def fitness(coeficients: list):

    data = get_data()
    return error(coeficients, data)

def migration_ratio(generations: int) -> int:
    """
    Calcula el ratio de migración en función del número de generaciones.
    
    Args:
        generations (int): Número total de generaciones.
    
    Returns:
        int: Ratio de migración.
    """
    return int(rnd.uniform(log(generations), sqrt(generations)))

def measure(func):
    """
    Decorador para medir el tiempo de ejecución y el uso de memoria de una función.
    
    Args:
        func (function): Función a medir.
    
    Returns:
        function: Función decorada que devuelve un diccionario con:
                  - 'solution': Resultado de la función original.
                  - 'time': Tiempo de ejecución en segundos.
                  - 'memory': Pico de memoria utilizado en bytes.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tm.start()  # Inicia el rastreo de memoria
        start_time = time.perf_counter()  # Tiempo de inicio

        result = func(*args, **kwargs)  # Ejecuta la función

        end_time = time.perf_counter()  # Tiempo de fin
        peak = tm.get_traced_memory()[1]  # Obtiene el uso máximo de memoria
        tm.stop()  # Detiene el rastreo de memoria
        return {'solution': result, 'time': end_time - start_time, 'memory': peak}
    return wrapper