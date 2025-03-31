import numpy as np
from scipy.optimize import curve_fit

def polynomial_function(x, a, b, c, d, e, g, h, i):
    """
    Función polinómica de la forma:
    f(x) = e^(a) + b*x + c*x^2 + d*x^3 + e*x^4 + g*x^5 + h*x^6 + i*x^7

    Args:
        x (np.array): Valores de entrada.
        a, b, c, d, e, g, h, i (float): Coeficientes del polinomio.

    Returns:
        np.array: Valores calculados de la función polinómica.
    """
    return np.exp(a) + b*x + c*x**2 + d*x**3 + e*x**4 + g*x**5 + h*x**6 + i*x**7


def fit_polynomial(x, y, initial_guess=None):
    """
    Ajusta los coeficientes de un polinomio a los datos dados utilizando `curve_fit`.

    Args:
        x (np.array): Valores de entrada (x).
        y (np.array): Valores de salida (y).
        initial_guess (list, optional): Valores iniciales para los coeficientes. 
                                         Si no se proporciona, se inicializan en 0.

    Returns:
        tuple: Una tupla con:
            - params (np.array): Coeficientes ajustados.
            - covariance (np.array): Matriz de covarianza de los coeficientes.
    """
    if initial_guess is None:
        initial_guess = [0] * 8  # Por defecto, inicializamos todos los coeficientes en 0

    params, covariance = curve_fit(polynomial_function, x, y, p0=initial_guess)
    return params, covariance