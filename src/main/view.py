import streamlit as st
import algorithms.genetic as gnc
import tools.utils as utils
import pandas as pd
import numpy as np
import functions.selection as select
import functions.crossing as cross
import functions.mutation as mutate
import functions.replacement as replace
import algorithms.regression as regression

# ============================
# Funciones auxiliares
# ============================

def plot_function(coefs):
    """
    Plotea la función ajustada con los coeficientes dados como un line chart.
    """
    x = np.linspace(-2, 2, 100)
    y = utils.f(x, coefs)
    chart_data = pd.DataFrame({"y": y})
    st.line_chart(chart_data, color=["rgb(255, 0, 0)"])

def plot_predictions(predictions, actuals):
    """
    Plotea las predicciones y los valores reales como un scatter chart.
    """
    chart_data = pd.DataFrame({
        "Predicciones": predictions,
        "Valores Reales": actuals
    })
    st.scatter_chart(chart_data, color=["rgb(255, 0, 0)", "rgb(0, 0, 255)"], use_container_width=True)

def display_coefficients(coefs):
    """
    Muestra los coeficientes encontrados en formato LaTeX.
    """
    if len(coefs) < 8:
        st.error("El número de coeficientes es menor a 8. No se pueden mostrar todos los valores.")
        return

    st.latex(f"""
        \\begin{{array}}{{ll}}
                 a & = {coefs[0]} \\\\
                 b & = {coefs[1]} \\\\
                 c & = {coefs[2]} \\\\
                 d & = {coefs[3]} \\\\
                 e & = {coefs[4]} \\\\
                 f & = {coefs[5]} \\\\
                 g & = {coefs[6]} \\\\
                 h & = {coefs[7]} \\\\
        \\end{{array}}
        """)

# ============================
# Interfaz de usuario
# ============================

# Título y descripción
st.title("Optimización Genética con Modelo de Islas")
st.markdown("""
Esta aplicación utiliza un algoritmo genético basado en el modelo de islas para resolver problemas de optimización.
El objetivo es encontrar los coeficientes que mejor se ajusten a los datos, minimizando el error cuadrático medio (MSE).
""")

# Mostrar los datos
st.subheader("Datos de entrada")
data = utils.data(compact=True)

data = np.sort(data, axis=0)
frame = pd.DataFrame(data, columns=["x", "y"])
st.dataframe(frame)

# ============================
# Configuración del algoritmo
# ============================

st.sidebar.header("Configuración del Algoritmo")
num_islands = st.sidebar.number_input("Número de Islas", min_value=1, max_value=10, value=5, step=1)
pop_size = st.sidebar.number_input("Tamaño de la Población por Isla", min_value=1, max_value=1000, value=20, step=10)
generations = st.sidebar.number_input("Número de Generaciones", min_value=1, max_value=1000, value=100, step=10)
num_coef = 8

st.sidebar.subheader("Operadores Genéticos")
selection_method = st.sidebar.selectbox("Método de Selección", ["Torneo Binario", "Ruleta", "Ranking"])
crossover_method = st.sidebar.selectbox("Método de Cruce", ["Cruce Aritmético", "Cruce de Un Punto", "Cruce Uniforme"])
mutation_method = st.sidebar.selectbox("Método de Mutación", ["Mutación Gaussiana", "Mutación Uniforme"])
replacement_method = st.sidebar.selectbox("Método de Reemplazo", ["Elitismo", "Reemplazo Generacional Completo"])

# ============================
# Ejecución del algoritmo
# ============================


st.header("Ejecución del Algoritmo Genético")
ag_result = None

if st.button("Ejecutar Algoritmo"):
    st.subheader("Ejecución del Algoritmo")
    with st.spinner("Ejecutando el modelo de islas..."):
        # Mapear los operadores seleccionados
        selection = {
            "Torneo Binario": select.tournament,
        }[selection_method]

        crossover = {
            "Cruce Aritmético": cross.arithmetic,
        }[crossover_method]

        mutation = {
            "Mutación Gaussiana": mutate.gaussian,
        }[mutation_method]

        replacement = {
            "Elitismo": replace.elitism,
        }[replacement_method]

        # Ejecutar el algoritmo genético
        result = gnc.island_optimization(
            num_islands, pop_size, generations, num_coef,
            selection, crossover, mutation, replacement, utils.fitness
        )

        st.success("Optimización completada.")

        # Mostrar resultados
        st.subheader("Resultados")
        coefs = result['solution']['coefficients']
        st.write("Coeficientes encontrados:")
        display_coefficients(coefs)

        st.latex(f"""
        \\text{{MSE}} = {result['solution']['error']:.4f} 
        """)

        # Mostrar memoria y tiempo de ejecución
        st.latex(f"""
        \\text{{Memoria utilizada}} = {result['memory']} \\text{{ bytes}}
        """)

        st.latex(f"""
        \\text{{Tiempo de ejecución}} = {result['time']:.4f} \\text{{ segundos}}
        """)

        # Gráficas
        st.subheader("Comparación de predicciones")
        x, y = utils.data()
        predictions = utils.f(x, coefs)
        plot_predictions(predictions, y)

        st.subheader("Gráfica de la función ajustada")
        plot_function(coefs)

# ============================
# Comparación con regresión lineal
# ============================

st.header("Comparación con el Modelo de Regresión Lineal")
st.markdown("""
El modelo de regresión lineal se ajusta a los datos utilizando la librería `scikit-learn`.
El objetivo es comparar el rendimiento del algoritmo genético con el modelo de regresión lineal tradicional.
""")

if st.button("Ejecutar Regresión Lineal"):
    st.subheader("Ejecución del Modelo de Regresión Lineal")
    with st.spinner("Ajustando el modelo de regresión lineal..."):
        x, y = utils.data()
        params, covariance = regression.fit_polynomial(x, y)

        st.success("Regresión lineal completada.")

        # Mostrar resultados
        st.subheader("Resultados de la Regresión Lineal")
        display_coefficients(params)

        # Gráficas
        st.subheader("Gráfica de la función ajustada")
        plot_function(params)
        st.subheader("Comparación de predicciones")
        predictions = regression.polynomial_function(x, *params)
        plot_predictions(predictions, y)
        st.latex(f"""
        \\text{{MSE}} = {utils.error(params, x, y):.4f} 
        """)