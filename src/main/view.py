import streamlit as st
import pandas as pd
import numpy as np

import algorithms.genetic as gnc
import algorithms.regression as regression

import tools.utils as utils
import functions.selection as select
import functions.crossing as cross
import functions.mutation as mutate
import functions.replacement as replace

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

st.header("Configuración del Algoritmo")
num_islands = st.number_input("Número de Islas", min_value=1, max_value=10, value=5, step=1)
pop_size = st.number_input("Tamaño de la Población por Isla", min_value=1, max_value=1000, value=20, step=10)
generations = st.number_input("Número de Generaciones", min_value=1, max_value=1000, value=100, step=10)
num_coef = 8

st.subheader("Operadores Genéticos")
st.markdown("""#### Operadores de selección""")
selection_method = st.selectbox("Método de Selección", list(select.selections().keys()))
selection = select.selections()[selection_method]

if selection_method == "Torneo Binario":
    k = st.number_input("Número de individuos para el torneo (k)", min_value=2, max_value=pop_size, value=2, step=1)
    selection = selection(k)
elif selection_method == "Aleatorio":
    n = st.number_input("Número de individuos a seleccionar", min_value=1, max_value=pop_size, value=1, step=1)
    selection = selection(n)
elif selection_method == "Ruleta":
    n = st.number_input("Número de individuos a seleccionar", min_value=1, max_value=pop_size, value=1, step=1)
    selection = selection(n) 
elif selection_method == "Emparejamiento variado inverso":
    k = st.number_input("Número de individuos en el subconjunto de candidatos", min_value=2, mac_value=pop_size, value=2, step=1)
    selection = selection(k)

st.markdown("""#### Operadores de cruce""")
crossover_method = st.selectbox("Método de Cruce", list(cross.crossings().keys()))

crossover = cross.crossings()[crossover_method]


st.markdown("""#### Operadores de mutación""")
mutation_method = st.selectbox("Método de Mutación", list(mutate.mutations().keys()))

mutation = mutate.mutations()[mutation_method]

if mutation_method == "Mutación Gaussiana":
    sigma = st.number_input("Desviación estándar", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    mutation = mutation(sigma)
else:
    mutation = mutation()

mutation_rate = st.number_input("Tasa de mutación", min_value=0.0, max_value=1.0, value=0.1, step=0.01)


st.markdown("""#### Operadores de reemplazo""")
replacement_method = st.selectbox("Método de Reemplazo", list(replace.replacements().keys()))

replacement = replace.replacements()[replacement_method]

# ============================
# Ejecución del algoritmo
# ============================

st.header("Ejecución del Algoritmo Genético")

# Inicializar variables en session_state
if "ag_result" not in st.session_state:
    st.session_state.ag_result = None

if st.button("Ejecutar Algoritmo"):
    st.subheader("Ejecución del Algoritmo")
    with st.spinner("Ejecutando el modelo de islas..."):
        # Mapear los operadores seleccionados
        
        result = gnc.island_optimization(
            num_islands, pop_size, generations, num_coef,
            selection, 
            crossover, 
            mutation, 
            mutation_rate,
            replacement, 
            utils.fitness
        )

        st.success("Optimización completada.")
        st.session_state.ag_result = result  # Guardar resultados en session_state

# Mostrar resultados si existen
if st.session_state.ag_result:
    result = st.session_state.ag_result
    st.subheader("Resultados")
    coefs = result['coefficients']
    st.write("Coeficientes encontrados:")
    display_coefficients(coefs)

    st.latex(f"""
    \\text{{MSE}} = {result['error']:.4f} 
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

# Inicializar variables en session_state
if "regression_result" not in st.session_state:
    st.session_state.regression_result = None

if st.button("Ejecutar Regresión Lineal"):
    st.subheader("Ejecución del Modelo de Regresión Lineal")
    with st.spinner("Ajustando el modelo de regresión lineal..."):
        x, y = utils.data()
        params, covariance = regression.fit_polynomial(x, y)

        st.success("Regresión lineal completada.")
        st.session_state.regression_result = {"params": params, "x": x, "y": y}  # Guardar resultados

# Mostrar resultados si existen
if st.session_state.regression_result:
    params = st.session_state.regression_result["params"]
    x = st.session_state.regression_result["x"]
    y = st.session_state.regression_result["y"]

    st.subheader("Resultados de la Regresión Lineal")
    display_coefficients(params)
    st.latex(f"""
    \\text{{MSE}} = {utils.error(params, x, y):.4f} 
    """)
    # Gráficas
    st.subheader("Gráfica de la función ajustada")
    plot_function(params)
    st.subheader("Comparación de predicciones")
    predictions = regression.polynomial_function(x, *params)
    plot_predictions(predictions, y)


    # ============================
    # Comparación de Resultados
    # ============================

    st.header("Comparación de Resultados")
    st.markdown("""
    En esta sección se comparan los resultados obtenidos por el algoritmo genético y el modelo de regresión lineal.
    Se evalúan los coeficientes encontrados, el error cuadrático medio (MSE) y las gráficas de predicción.
    """)

    if st.session_state.ag_result and st.session_state.regression_result:
        # Obtener resultados del algoritmo genético
        ag_coefs = st.session_state.ag_result['coefficients']
        ag_mse = st.session_state.ag_result['error']

        # Obtener resultados de la regresión lineal
        reg_coefs = st.session_state.regression_result["params"]
        reg_mse = utils.error(reg_coefs, st.session_state.regression_result["x"], st.session_state.regression_result["y"])

        # Mostrar comparación de MSE
        st.subheader("Comparación de MSE")
        st.write(f"**Algoritmo Genético:** {ag_mse:.4f}")
        st.write(f"**Regresión Lineal:** {reg_mse:.4f}")

        # Comparación de coeficientes
        st.subheader("Comparación de Coeficientes")
        st.write("**Coeficientes del Algoritmo Genético:**")
        display_coefficients(ag_coefs)
        st.write("**Coeficientes de la Regresión Lineal:**")
        display_coefficients(reg_coefs)

        # Comparación de gráficas
        # Comparación de gráficas
        st.subheader("Comparación de Gráficas")
        st.markdown("**Predicciones del Algoritmo Genético vs. Regresión Lineal**")

        # Obtener datos
        x, y = utils.data()
        ag_predictions = utils.f(x, ag_coefs)
        reg_predictions = regression.polynomial_function(x, *reg_coefs)

        # Crear el DataFrame con colores consistentes
        comparison_data = pd.DataFrame({"x": x}).set_index("x")
        comparison_data["Valores Reales"] = y

        show_ag = st.checkbox("Mostrar valores del algoritmo genético", value=True)
        show_linear = st.checkbox("Mostrar valores de la regresión lineal", value=True)

        colors = ["rgb(0, 255, 0)"]
        if show_ag:
            comparison_data["Predicciones AG"] = ag_predictions

        if show_linear:
            comparison_data["Predicciones RL"] = reg_predictions


        if show_ag and show_linear:
            colors = ["rgb(255, 0, 0)", "rgb(0, 0, 255)"] + colors
        elif show_ag:
            colors = ["rgb(255, 0, 0)"] + colors
        elif show_linear:
            colors = ["rgb(0, 0, 255)"] + colors

        elif not show_ag and not show_linear:
            st.warning("Por favor, seleccione al menos un método para mostrar los valores de predicción.")
        # Graficar


        st.line_chart(comparison_data, use_container_width=True, color=colors)
    else:
        st.warning("Por favor, ejecute ambos métodos para realizar la comparación.")
