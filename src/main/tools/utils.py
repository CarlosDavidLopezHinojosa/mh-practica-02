import time
import tracemalloc as tm
import numpy as np
import json
from typing import Literal

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
    # print(coeficients[0])
    c0 = coeficients[0]
    with np.errstate(over='ignore'):  # Ignora advertencias de desbordamiento
        try:
            exp_value = np.exp(coeficients[0])
            if np.isinf(exp_value):
                raise OverflowError("El valor calculado es infinito.")
        except OverflowError:
            # print("OverflowError: coeficients[0] es demasiado grande para calcular exp.")
            exp_value = 3.4903
            c0 = np.log(exp_value)
    coeficients[0] = exp_value
    value = np.polyval(coeficients[::-1], x)
    coeficients[0] = c0
    return value

def error(coeficients: np.array, x: np.array, y: np.array) -> float:
    """
    Calcula el error cuadrático total entre los valores estimados y los valores reales.
    
    Args:
        coeficients (np.array): Coeficientes del modelo.
        x (np.array): Valores de entrada.
        y (np.array): Valores reales esperados.
    
    Returns:
        float: Error cuadrático total.
    """
    predictions = f(x, coeficients)
    predictions = np.clip(predictions, -1e10, 1e10)  # Limitar valores extremos en las predicciones
    return np.mean((predictions - y) ** 2)

def fitness(coeficients: np.array):

    x, y = data()
    return error(coeficients, x, y)

def instant() -> float:
    """
    Devuelve el tiempo actual en segundos desde la época (epoch).
    
    Returns:
        float: Tiempo actual en segundos.
    """
    return time.perf_counter()

def memory() -> float:
    """
    Devuelve el uso máximo de memoria en bytes.
    
    Returns:
        float: Uso máximo de memoria en bytes.
    """
    return tm.get_traced_memory()[0]

def memstart():
    """
    Inicia el rastreo de memoria.
    """
    tm.start()

def memstop():
    """
    Detiene el rastreo de memoria.
    """
    tm.stop()


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
        memstart()  # Inicia el rastreo de memoria
        start_time = time.perf_counter()  # Tiempo de inicio
        result = {}
        try:
            result.update(func(*args, **kwargs))  # Ejecuta la función
        finally:
            end_time = time.perf_counter()  # Tiempo de fin
            peak = tm.get_traced_memory()[1]  # Obtiene el uso máximo de memoria
            memstop()  # Detiene el rastreo de memoria
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
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load(filename):
    """
    Carga datos desde un archivo utilizando pickle.
    
    Args:
        filename (str): Nombre del archivo desde donde se cargarán los datos.
    
    Returns:
        any: Datos cargados desde el archivo.
    """
    with open(filename, 'r') as f:
        return json.load(f)

def best_algorithm_config(mode: bool = False):
    """
    Devuelve la mejor configuración de algoritmo para el problema de optimización.

        - Tamano de la población: 23
        - Generaciones: 143
        - Selección: Torneo Binario con k=7
        - Cruce: BLX
        - Mutación: Gaussiana con sigma=0.05
        - Reemplazo: Elitismo

    Args:
        mode (bool): Modo de ejecución. Si es True, mide la convergencia de los operadores.
    
    Returns:
        dict: Diccionario con la mejor configuración de algoritmo.
    """

    import functions.selection as select
    import functions.crossing as cross
    import functions.mutation as mutate
    import functions.replacement as replace
    return {
        'island': population(23, 8),
        'pop_size': 23,
        'generations': 143,
        'select': select.tournament(7, fitness, mode),
        'cross': cross.BLX(fitness, mode),
        'mutate': mutate.gaussian(0.05, fitness, mode),
        'replace': replace.elitism(fitness, mode),
        'fitness': fitness,
    }