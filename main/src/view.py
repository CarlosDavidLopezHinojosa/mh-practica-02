import streamlit as st
import numpy as np

import algorithms.genetic as gnc
import algorithms.regression as regression

import functions.selection as select
import functions.crossing as cross
import functions.mutation as mutate
import functions.replacement as replace

import tools.utils as utils
import tools.stats as stats
import tools.plot as plot


# ============================
# Funciones auxiliares
# ============================

def configure_algorithm():
    """
    Configura los parámetros del algoritmo genético y selecciona los operadores.
    Returns:
        dict: Configuración del algoritmo genético.
    """
    st.header("Configuración del Algoritmo")
    config = {}

    # Parámetros generales
    config["num_islands"] = st.number_input("Número de Islas", min_value=1, max_value=10, value=5, step=1)
    config["pop_size"] = st.number_input("Tamaño de la Población por Isla", min_value=1, max_value=1000, value=20, step=10)
    config["generations"] = st.number_input("Número de Generaciones", min_value=1, max_value=1000, value=100, step=10)
    config["num_coef"] = 8

    # Operadores genéticos
    st.subheader("Operadores Genéticos")

    # Selección
    st.markdown("#### Operadores de selección")
    selection_method = st.selectbox("Método de Selección", list(select.selections().keys()))
    selection = select.selections()[selection_method]
    if selection_method == "Torneo Binario":
        k = st.number_input("Número de individuos para el torneo (k)", min_value=2, max_value=config["pop_size"], value=2, step=1)
        selection = selection(k, utils.fitness)
    elif selection_method == "Ruleta":
        n = st.number_input("Número de individuos para la selección (n)", min_value=2, max_value=config["pop_size"], value=2, step=1)
        selection = selection(n, utils.fitness)
    elif selection_method == "Emparejamiento variado inverso":
        k = st.number_input("Número de individuos para la selección (k)", min_value=2, max_value=config["pop_size"], value=2, step=1)
        selection = selection(k, utils.fitness)
    elif selection_method == "Aleatorio":
        n = st.number_input("Número de individuos a seleccionar", min_value=1, max_value=config["pop_size"], value=1, step=1)
        selection = selection(n, utils.fitness)
    else:
        selection = selection()
    config["selection"] = selection

    # Cruce
    st.markdown("#### Operadores de cruce")
    crossover_method = st.selectbox("Método de Cruce", list(cross.crossings().keys()))
    config["crossover"] = cross.crossings()[crossover_method](utils.fitness)

    # Mutación
    st.markdown("#### Operadores de mutación")
    mutation_method = st.selectbox("Método de Mutación", list(mutate.mutations().keys()))
    mutation = mutate.mutations()[mutation_method]
    if mutation_method == "Mutación Gaussiana":
        sigma = st.number_input("Desviación estándar", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        mutation = mutation(sigma, utils.fitness)

    elif mutation_method == "Mutación No Uniforme":
       mutation = mutation(config['generations'], utils.fitness)
    else:
        mutation = mutation(utils.fitness) # Clase por defecto sin parametros

    config["mutation"] = mutation
    # config["mutation_rate"] = st.number_input("Tasa de mutación", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    # Reemplazo
    st.markdown("#### Operadores de reemplazo")
    replacement_method = st.selectbox("Método de Reemplazo", list(replace.replacements().keys()))
    replacement = replace.replacements()[replacement_method]

    if replacement_method == "Reemplazar al peor de la población":
        replacement = replacement(utils.fitness)
    elif replacement_method == "Torneo restringido":
        n = st.number_input("Número de individuos para el torneo (n)", min_value=1, max_value=config["pop_size"], value=1, step=1)
        replacement = replacement(n, utils.fitness)
    elif replacement_method == "Peor entre semejantes":
        n = st.number_input("Número de individuos más parecidos entre los que reemplazar (n)", min_value=1, max_value=config["pop_size"], value=1, step=1)
        replacement = replacement(n, utils.fitness)
    elif replacement_method == "Elitismo":
        replacement = replacement(utils.fitness)
    else:
        replacement = replacement(utils.fitness)

    config["replacement"] = replacement

    return config

def execute_genetic_algorithm(config):
    """
    Ejecuta el algoritmo genético con la configuración proporcionada.
    Args:
        config (dict): Configuración del algoritmo genético.
    Returns:
        dict: Resultados del algoritmo genético.
    """
    st.header("Ejecución del Algoritmo Genético")
    if st.button("Ejecutar Algoritmo"):
        st.subheader("Ejecución del Algoritmo")
        with st.spinner("Ejecutando el modelo de islas..."):
            result = gnc.island_optimization(
                config["num_islands"], config["pop_size"], config["generations"], config["num_coef"],
                config["selection"], config["crossover"], config["mutation"],
                config["replacement"], utils.fitness
            )
            st.success("Optimización completada.")
            st.session_state['AG'] = result
            return result
    return None

def plot_function(coeffs):
    """
    Plotea la función ajustada con los coeficientes dados como un line chart.
    """
    x = np.linspace(-2, 2, 100)
    y = utils.f(x, coeffs)
    st.line_chart({"y": y}, color=["rgb(255, 0, 0)"], use_container_width=True)

def plot_predictions(predictions, actuals):
    """
    Plotea las predicciones y los valores reales como un scatter chart.
    """
    chart_data = {
        "Predicciones": predictions,
        "Valores Reales": actuals
    }
    st.scatter_chart(chart_data, color=["rgb(255, 0, 0)", "rgb(0, 0, 255)"], use_container_width=True)

def display_coefficients(coeffs):
    """
    Muestra los coeficientes encontrados en formato LaTeX.
    """
    if len(coeffs) < 8:
        st.error("El número de coeficientes es menor a 8. No se pueden mostrar todos los valores.")
        return

    st.latex(f"""
        \\begin{{array}}{{ll}}
                 a & = {coeffs[0]} \\\\
                 b & = {coeffs[1]} \\\\
                 c & = {coeffs[2]} \\\\
                 d & = {coeffs[3]} \\\\
                 e & = {coeffs[4]} \\\\
                 f & = {coeffs[5]} \\\\
                 g & = {coeffs[6]} \\\\
                 h & = {coeffs[7]} \\\\
        \\end{{array}}
        """)

def display_results(result):
    """
    Muestra los resultados del algoritmo genético.
    Args:
        result (dict): Resultados del algoritmo genético.
    """
    if result:
        st.subheader("Resultados")
        coeffs = result['coefficients']
        st.write("Coeficientes encontrados:")
        display_coefficients(coeffs)

        st.latex(f"\\text{{MSE}} = {result['error']:.4f}")

        if 'memory' in result:
            st.latex(f"\\text{{Memoria utilizada}} = {result['memory']} \\text{{ bytes}}")
        if 'time' in result:
            st.latex(f"\\text{{Tiempo de ejecución}} = {result['time']:.4f} \\text{{ segundos}}")

        # Gráficas
        st.subheader("Comparación de predicciones")
        x, y = utils.data()
        predictions = utils.f(x, coeffs)
        plot_predictions(predictions, y)

        st.subheader("Gráfica de la función ajustada")
        plot_function(coeffs)

def execute_regression():
    """
    Ejecuta el modelo de regresión lineal y muestra los resultados.
    Returns:
        dict: Resultados del modelo de regresión lineal.
    """
    st.header("Ejecución del Modelo de Regresión Lineal")
    if st.button("Ejecutar Regresión Lineal"):
        st.subheader("Ejecución del Modelo de Regresión Lineal")
        with st.spinner("Ajustando el modelo de regresión lineal..."):
            x, y = utils.data()
            coefficients, covariance = regression.fit_polynomial(x, y)
            st.success("Regresión lineal completada.")
            result = {"coefficients": coefficients, 'error': utils.error(coefficients, x, y)}
            st.session_state['RL'] = result
            return result
    return None

def compare_results(ag_result, regression_result):
    """
    Compara los resultados del algoritmo genético y la regresión lineal.
    Args:
        ag_result (dict): Resultados del algoritmo genético.
        regression_result (dict): Resultados del modelo de regresión lineal.
    """
    if ag_result and regression_result:
        st.header("Comparación de Resultados")
        st.markdown("""
        En esta sección se comparan los resultados obtenidos por el algoritmo genético y el modelo de regresión lineal.
        Se evalúan los coeficientes encontrados, el error cuadrático medio (MSE) y las gráficas de predicción.
        """)
        x, y = utils.data()
        # Mostrar comparación de MSE
        ag_mse = ag_result['error']
        reg_mse = utils.error(regression_result["coefficients"], x, y)
        st.subheader("Comparación de MSE")
        st.write(f"**Algoritmo Genético:** {ag_mse:.4f}")
        st.write(f"**Regresión Lineal:** {reg_mse:.4f}")

        # Comparación de coeficientes
        st.subheader("Comparación de Coeficientes")
        st.write("**Coeficientes del Algoritmo Genético:**")
        display_coefficients(ag_result['coefficients'])
        st.write("**Coeficientes de la Regresión Lineal:**")
        display_coefficients(regression_result["coefficients"])

        # Comparación de gráficas
        st.subheader("Comparación de Gráficas")
        
        ag_predictions = utils.f(x, ag_result['coefficients'])
        reg_predictions = regression.polynomial_function(x, *regression_result["coefficients"])
        plot_predictions(ag_predictions, y)
        plot_predictions(reg_predictions, y)


# ============================
# Ejecución principal
# ============================

# Título y descripción
st.title("Optimización Genética con Modelo de Islas")
st.markdown("""
Esta aplicación utiliza un algoritmo genético basado en el modelo de islas para resolver problemas de optimización.
El objetivo es encontrar los coeficientes que mejor se ajusten a los datos, minimizando el error cuadrático medio (MSE).
            
Dichos coeficientes pertencen a la siguiente función:
            
$$f(x) = e^{a} + bx + cx^2 + dx^3 + ex^4 + fx^5 + gx^6 + hx^7$$
""")

# Mostrar los datos
st.subheader("Datos de entrada")
data = utils.data(compact=True)
data = np.sort(data, axis=0)
st.dataframe({"x": data[:, 0], "y": data[:, 1]}, use_container_width=True)

# Configuración del algoritmo
config = configure_algorithm()

# Ejecutar algoritmo genético
execute_genetic_algorithm(config)

# Mostrar resultados del algoritmo genético
if "AG" in st.session_state:
    display_results(st.session_state['AG'])

# Ejecutar regresión lineal
execute_regression()

# Mostrar resultados de la regresión lineal
if 'RL' in st.session_state:
    display_results(st.session_state['RL'])

# Comparar resultados
if "AG" in st.session_state and "RL" in st.session_state:
    compare_results(st.session_state['AG'], st.session_state['RL'])
SAVEPATH = "main/info/"

figs = plot.process_file(SAVEPATH + "mutations.json")

print(figs)

for name, fig in figs.items():
    st.subheader(name)
    st.plotly_chart(fig, use_container_width=True)