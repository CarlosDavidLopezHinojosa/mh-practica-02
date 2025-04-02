import time
import tracemalloc as tm
import numpy as np
import pickle as pkl

from functools import wraps

def coeficients(num_coefs: int) -> np.array:
    """
    Genera un array de coeficientes aleatorios.
    
    Args:
        num_coefs (int): Número de coeficientes a generar.
    
    Returns:
        np.array: Array de tamaño `n` con valores aleatorios entre 0 y 1.
    """
    return np.random.random(num_coefs)

def population(pop_size: int, num_coefs: int) -> np.array:
    """
    Genera una población inicial de individuos con coeficientes aleatorios.
    
    Args:
        pop_size (int): Tamaño de la población (número de individuos).
        num_coefs (int): Número de coeficientes por individuo.
    
    Returns:
        np.array: Matriz de tamaño `(pop_size, n)` con coeficientes aleatorios.
    """
    return np.random.random((pop_size, num_coefs))

def create_islands(num_island, pop_size, num_coefs):
    """
    Genera una lista de poblaciones para el modelo de islas.
    
    Args:
        num_island (int): Número de islas.
        pop_size (int): Tamaño de la población por isla.
        num_coefs (int): Número de coeficientes por individuo.
    
    Returns:
        np.array: Lista de poblaciones para cada isla.
    """
    return np.random.random((num_island, pop_size, num_coefs))

def data(compact: bool = False, lower: int = 0, upper: int = 15) -> np.array:
    """
    Obtiene un subconjunto de datos de prueba en el rango especificado.
    
    Args:
        lower (int, opcional): Índice inferior del subconjunto. Por defecto, 0.
        upper (int, opcional): Índice superior del subconjunto. Por defecto, 15.
    
    Returns:
        np.array: Matriz de datos en formato `[x, y]` con valores en el rango `[lower:upper]`.
    """
    data = np.array([
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
    ])

    np.random.shuffle(data)

    if compact:
        return data[lower:upper]
    
    x = data[lower:upper, 0]
    y = data[lower:upper, 1]
    return x, y

def f(x: float, coeficients: np.array) -> float:
    """
    Calcula el valor de una función basada en coeficientes polinomiales.
    
    La función tiene la forma:
        f(x) = exp(coeficients[0]) + coeficients[1]*x^1 + coeficients[2]*x^2 + ...
    
    Args:
        x (float): Valor de entrada.
        coeficients (np.array): Lista de coeficientes del modelo.
    
    Returns:
        float: Resultado de la función para el valor `x` dado.
    """
    return np.exp(coeficients[0]) + np.sum(coeficients[i] * x ** i for i in range(1, len(coeficients)))

def error(coeficients: np.array, x: np.array, y: np.array) -> float:
    """
    Calcula el error cuadrático total entre los valores estimados y los valores reales.
    
    Args:
        coeficients (np.array): Coeficientes del modelo.
        data (np.array): Matriz de datos en formato `[x, y]`, donde:
                         - `x` es el valor de entrada.
                         - `y` es el valor real esperado.
    
    Returns:
        float: Error cuadrático total.
    """
    return np.sum(np.square(f(x, coeficients) - y))

def fitness(coeficients: np.array):

    x, y = data()
    return error(coeficients, x, y)

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
        result = {}
        try:
            result.update(func(*args, **kwargs))  # Ejecuta la función
        finally:
            end_time = time.perf_counter()  # Tiempo de fin
            peak = tm.get_traced_memory()[1]  # Obtiene el uso máximo de memoria
            tm.stop()  # Detiene el rastreo de memoria
        result.update({'time': end_time - start_time, 'memory': peak})
        return result
    return wrapper

def save(data, filename):
    """
    Guarda datos en un archivo utilizando pickle.
    
    Args:
        data (any): Datos a guardar.
        filename (str): Nombre del archivo donde se guardarán los datos.
    """
    with open(filename, 'wb') as f:
        pkl.dump(data, f)

def load(filename):
    """
    Carga datos desde un archivo utilizando pickle.
    
    Args:
        filename (str): Nombre del archivo desde donde se cargarán los datos.
    
    Returns:
        any: Datos cargados desde el archivo.
    """
    with open(filename, 'rb') as f:
        return pkl.load(f)
