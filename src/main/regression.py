import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Definimos la fórmula de la regresión
def polynomial_function(x, a, b, c, d, e, g, h, i):
    """
    Función polinómica de la forma:
    f(x) = e^(a) + b*x + c*x^2 + d*x^3 + e*x^4 + g*x^5 + h*x^6 + i*x^7
    """
    return np.exp(a) + b*x + c*x**2 + d*x**3 + e*x**4 + g*x**5 + h*x**6 + i*x**7

# Datos de ejemplo (puedes reemplazarlos con tus datos reales)
x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
y_data = np.array([1.5, 2.3, 3.8, 5.1, 7.3, 9.8, 13.2, 17.5])

import utils

data = utils.get_data()

x_data = np.array([x for x, y in data])
y_data = np.array([y for x, y in data])

# Ajustamos los coeficientes usando curve_fit
initial_guess = [0, 0, 0, 0, 0, 0, 0, 0]  # Valores iniciales para los coeficientes
params, covariance = curve_fit(polynomial_function, x_data, y_data, p0=initial_guess)

# Extraemos los coeficientes ajustados
a, b, c, d, e, g, h, i = params

# Mostramos los resultados
print("Coeficientes ajustados:")
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"d = {d}")
print(f"e = {e}")
print(f"g = {g}")
print(f"h = {h}")
print(f"i = {i}")

# Graficamos los datos originales y la curva ajustada
x_fit = np.linspace(min(x_data), max(x_data), 500)
y_fit = polynomial_function(x_fit, a, b, c, d, e, g, h, i)

# Calculamos el error del modelo (error cuadrático medio)
mse = np.mean((y_data - polynomial_function(x_data, a, b, c, d, e, g, h, i))**2)
print(f"Error cuadrático medio (MSE): {mse}")

plt.scatter(x_data, y_data, label="Datos originales", color="red")
plt.plot(x_fit, y_fit, label="Curva ajustada", color="blue")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("Regresión Polinómica")
plt.show()