import random as rnd
import math
import numpy as np

def random_coeficients(n: int) -> list:
    """
    Genera un array de coeficientes aleatorios.
    Args:
        n (int): Número de coeficientes.
    Returns:
        list: Array con valores aleatorios entre 0 y 1.
    """
    return [rnd.random() for _ in range(n)]

def initial_population(pop_size: int, n: int) -> list:
    """
    Genera una población inicial de coeficientes aleatorios.
    Args:
        pop_size (int): Tamaño de la población.
        n (int): Número de coeficientes por individuo.
    Returns:
        list: Matriz de coeficientes aleatorios.
    """
    return [random_coeficients(n) for _ in range(pop_size)]

def get_data(lower: int = 0, upper: int = 15) -> list[list]:
    """
    Obtiene un subconjunto de datos de prueba.
    Args:
        lower (int, opcional): Índice inferior del subconjunto. Por defecto 0.
        upper (int, opcional): Índice superior del subconjunto. Por defecto 15.
    Returns:
        list[list]: Subconjunto de datos en el rango especificado.
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
    Calcula el valor de una función basada en coeficientes.
    Args:
        x (float): Valor de entrada.
        coeficients (list): Array de coeficientes.
    Returns:
        float: Resultado de la función.
    """
    return math.exp(coeficients[0]) + sum([coeficients[i] * x ** i for i in range(1, len(coeficients))])

data = get_data()
def error(coeficients: list) -> float:
    """
    Calcula el error cuadrático entre valores estimados y valores reales.
    Args:
        coeficients (list): Coeficientes del modelo.
        data (list of list): Datos en pares (x, y).
    Returns:
        float: Error cuadrático total.
    """
    global data
    return sum([(f(x, coeficients) - y) ** 2 for x, y in data])
