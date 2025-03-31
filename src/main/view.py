import streamlit as st

import algorithms.genetic as gnc
import tools.utils as utils

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import functions.selection as select
import functions.crossing as cross
import functions.mutation as mutate
import functions.replacement as replace


def plot_function(coefs):
    """
    Plotea la función ajustada con los coeficientes dados como un line chart.
    """
    x = np.linspace(-2, 2, 100)
    y = utils.f(x, coefs)
    chart_data = pd.DataFrame({
        "y": y
    })
    st.line_chart(chart_data, color=["rgba(255, 0, 0, 0.8)"])

def plot_predictions(predictions, actuals):
    """
    Plotea las predicciones y los valores reales como un scatter chart.
    """
    chart_data = pd.DataFrame({
        "Predicciones": predictions,
        "Valores Reales": actuals
    })

    st.scatter_chart(chart_data, color=["rgba(255, 0, 0, 0.8)", "rgba(0, 0, 255, 0.8)"])



# Título de la aplicación
st.title("Optimización Genética con Modelo de Islas")

# Descripción del proyecto
st.markdown("""
Esta aplicación utiliza un algoritmo genético basado en el modelo de islas para resolver problemas de optimización.
El modelo divide la población en subpoblaciones (islas) que evolucionan de forma independiente y ocasionalmente migran individuos entre ellas.
            
Queremos optimizar los coeficientes de un modelo de regresión lineal para minimizar el error cuadrático medio (MSE) en un conjunto de datos.
            
El objetivo es encontrar los coeficientes que mejor se ajusten a los datos, utilizando un enfoque evolutivo.

La función que queremos optimizar es la siguiente:
$$
f(x) = e^{a} + bx + cx^2 + dx^3 + ex^4 + fx^5 + gx^6 + hx^7
$$
            
Y debemos ajustar el modelo a los siguientes datos:
""")

d = utils.data(compact=True)
frame = pd.DataFrame(d, columns=["x", "y"])
st.dataframe(frame)

# Sección: Selección
st.header("Selección")
st.markdown("""
- **Torneo binario:** Selecciona aleatoriamente dos individuos y elige el mejor. Es simple y eficaz, asegurando una presión selectiva equilibrada.
- **Ruleta (proporcional a la aptitud):** Probabilidad de selección basada en la calidad de la solución. Puede ser útil, pero en problemas de optimización con valores de aptitud muy distintos puede ser menos efectivo.
- **Ranking:** Ordena los individuos por aptitud y asigna probabilidades de selección proporcionales a su posición. Es una opción más estable que la ruleta.

**Mejor opción recomendada:** Torneo binario, ya que mantiene un buen equilibrio entre explotación y exploración.
""")
st.code("""
# Selección: Torneo binario

""", language="python")

# Sección: Cruce
st.header("Cruce")
st.markdown("""
- **Cruce de un punto o multipunto:** Intercambia segmentos de los cromosomas entre los padres en una o varias posiciones fijas.
- **Cruce uniforme:** Mezcla genes de ambos padres con una probabilidad del $50\\%$. Mantiene diversidad pero puede perder estructuras buenas.
- **Cruce aritmético (promediado):** Genera un hijo tomando un promedio ponderado de los valores de los padres, útil en problemas con variables continuas.

**Mejor opción recomendada:** Cruce aritmético, ya que el problema requiere optimización de valores continuos y este operador favorece una mejor interpolación entre soluciones.
""")
st.code("""
# Cruce: Cruce aritmético

""", language="python")

# Sección: Mutación
st.header("Mutación")
st.markdown("""
- **Mutación gaussiana:** Añade ruido gaussiano a los genes para realizar pequeñas variaciones en los coeficientes.
- **Mutación uniforme:** Reemplaza valores por otros dentro de un rango aleatorio.
- **Mutación por perturbación adaptativa:** Reduce o aumenta la magnitud de la mutación según la etapa de la evolución.

**Mejor opción recomendada:** Mutación gaussiana, ya que permite pequeños ajustes en los coeficientes sin alterar demasiado la convergencia.
""")
st.code("""
# Mutación: Mutación gaussiana

""", language="python")

# Sección: Reemplazo
st.header("Reemplazo")
st.markdown("""
- **Reemplazo generacional completo:** Sustituye toda la población por los hijos generados.
- **Elitismo:** Mantiene los mejores individuos de la generación anterior.
- **Reemplazo estacionario:** Solo reemplaza una parte de la población en cada iteración.

**Mejor opción recomendada:** Elitismo combinado con reemplazo estacionario, ya que asegura la retención de las mejores soluciones mientras introduce diversidad.
""")
st.code("""
# Reemplazo: Elitismo


""", language="python")

# Sección: Algoritmo Genético (Optimización)
st.header("Algoritmo Genético (Optimización)")
st.markdown("""
El algoritmo genético basado en el modelo de islas divide la población en subpoblaciones (islas) que evolucionan de forma independiente. 
Cada cierto número de generaciones, los mejores individuos migran entre islas para mantener la diversidad genética.
""")
st.code("""
def genetic_function_optimization(island: np.array, pop_size: int, 
                                  generations: int, select: callable, 
                                  cross: callable, mutate: callable, 
                                  replace: callable, fitness: callable) -> dict:

    for _ in range(generations):
        new_island = np.empty((pop_size, island.shape[1]), dtype=island.dtype)
        for i in range(pop_size // 2):
            p1, p2 = select(island, fitness), select(island, fitness)
            while p1 is p2:
                p2 = select(island, fitness)

            ch1, ch2 = cross(p1, p2), cross(p2, p1)
            ch1, ch2 = mutate(ch1), mutate(ch2)
            new_island[i*2], new_island[i*2+1] = ch1, ch2

        replace(island, new_island, fitness)

    solution = min(island, key=fitness)
    return {'coefficients': solution, 'error': fitness(solution)}
""", language="python")

# Parámetros de entrada
st.header("Parámetros del Algoritmo")
num_islands = st.number_input("Número de Islas", min_value=1, max_value=10, value=5, step=1)
pop_size = st.number_input("Tamaño de la Población por Isla", min_value=1, max_value=1000, value=20, step=1)
generations = st.number_input("Número de Generaciones", min_value=1, max_value=1000, value=100, step=10)
num_coef = 8

# Botón para ejecutar el algoritmo
if st.button("Ejecutar Algoritmo"):
    with st.spinner("Ejecutando el modelo de islas..."):

        if __name__ == '__main__':
            result = gnc.island_optimization(
            num_islands, pop_size, generations, num_coef,
            select.tournament, 
            cross.arithmetic, 
            mutate.gaussian, 
            replace.elitism, 
            utils.fitness
            )
            st.success("Optimización completada.")
            st.subheader("Resultado")
            coefs = result['solution']['coefficients']
            st.write("Coeficientes encontrados:")

            st.latex(f"""
                    a = {coefs[0]} \\\\
                    b = {coefs[1]} \\\\
                    c = {coefs[2]} \\\\
                    d = {coefs[3]} \\\\
                    e = {coefs[4]} \\\\
                    f = {coefs[5]} \\\\
                    g = {coefs[6]} \\\\
                    h = {coefs[7]}
                     """)
            st.write("Error cuadrático medio (MSE):", result['solution']['error'])

            st.write("Memoria utilizada:", result['memory'], "bytes")
            st.write("Tiempo de ejecución:", result['time'], "segundos")

            st.subheader("Gráfica de la función ajustada")


            x, y = utils.data()
            plot_predictions(predictions=utils.f(x, coefs), actuals=y)
        
            st.subheader("La función predicha es:")
            plot_function(result['solution']['coefficients'])

st.header("Comparación con el modelo de regresión lineal")

st.markdown("""
El modelo de regresión lineal se ajusta a los datos utilizando la librería `scikit-learn`.
El objetivo es comparar el rendimiento del algoritmo genético con el modelo de regresión lineal tradicional.
""")
