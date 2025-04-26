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
from deap import base, creator, tools, algorithms
import time
import random


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

    # Opción para ejecución en paralelo
    config["parallel"] = st.checkbox("Ejecutar en paralelo", value=False)

    # Opción para medir convergencia
    config["measure_convergence"] = st.checkbox("Medir convergencia de los operadores", value=False)

    # Selección
    st.markdown("#### Operadores de selección")
    selection_method = st.selectbox("Método de Selección", list(select.selections().keys()))
    selection = select.selections()[selection_method]
    if selection_method == "Torneo Binario":
        k = st.number_input("Número de individuos para el torneo (k)", min_value=2, max_value=config["pop_size"], value=2, step=1)
        selection = selection(k, utils.fitness, config["measure_convergence"])
    elif selection_method == "Ruleta":
        n = st.number_input("Número de individuos para la selección (n)", min_value=2, max_value=config["pop_size"], value=2, step=1)
        selection = selection(n, utils.fitness, config["measure_convergence"])
    elif selection_method == "Emparejamiento variado inverso":
        k = st.number_input("Número de individuos para la selección (k)", min_value=2, max_value=config["pop_size"], value=2, step=1)
        selection = selection(k, utils.fitness, config["measure_convergence"])
    elif selection_method == "Aleatorio":
        n = st.number_input("Número de individuos a seleccionar", min_value=1, max_value=config["pop_size"], value=1, step=1)
        selection = selection(n, utils.fitness, config["measure_convergence"])
    else:
        selection = selection()
    config["selection"] = selection

    # Cruce
    st.markdown("#### Operadores de cruce")
    crossover_method = st.selectbox("Método de Cruce", list(cross.crossings().keys()))
    config["crossover"] = cross.crossings()[crossover_method](utils.fitness, config["measure_convergence"])

    # Mutación
    st.markdown("#### Operadores de mutación")
    mutation_method = st.selectbox("Método de Mutación", list(mutate.mutations().keys()))
    mutation = mutate.mutations()[mutation_method]
    if mutation_method == "Mutación Gaussiana":
        sigma = st.number_input("Desviación estándar", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        mutation = mutation(sigma, utils.fitness, config["measure_convergence"])
    else:
        mutation = mutation(utils.fitness, config["measure_convergence"]) # Clase por defecto sin parametros

    config["mutation"] = mutation
    # config["mutation_rate"] = st.number_input("Tasa de mutación", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    # Reemplazo
    st.markdown("#### Operadores de reemplazo")
    replacement_method = st.selectbox("Método de Reemplazo", list(replace.replacements().keys()))
    replacement = replace.replacements()[replacement_method]

    if replacement_method == "Reemplazar al peor de la población":
        replacement = replacement(utils.fitness, config["measure_convergence"])
    elif replacement_method == "Torneo restringido":
        n = st.number_input("Número de individuos para el torneo (n)", min_value=1, max_value=config["pop_size"], value=1, step=1)
        replacement = replacement(n, utils.fitness, config["measure_convergence"])
    elif replacement_method == "Peor entre semejantes":
        n = st.number_input("Número de individuos más parecidos entre los que reemplazar (n)", min_value=1, max_value=config["pop_size"], value=1, step=1)
        replacement = replacement(n, utils.fitness, config["measure_convergence"])
    elif replacement_method == "Elitismo":
        replacement = replacement(utils.fitness, config["measure_convergence"])
    else:
        replacement = replacement(utils.fitness, config["measure_convergence"])

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
            
            # Si 'parallel' es True, ejecutar en paralelo usando island_optimization
            if config["parallel"]:
                result = gnc.island_optimization(
                    config["num_islands"], config["pop_size"], config["generations"], config["num_coef"],
                    config["selection"], config["crossover"], config["mutation"],
                    config["replacement"], utils.fitness
                )
            else:
                # Si 'parallel' es False, ejecutar de manera secuencial usando genetic_function_optimization
                result = gnc.genetic_function_optimization(
                    utils.population(config["pop_size"], config["num_coef"]),
                    config["pop_size"], config["generations"],
                    config["selection"], config["crossover"],
                    config["mutation"], config["replacement"], utils.fitness
                )
            
            st.success("Optimización completada.")
            st.session_state['AG'] = result

            # Mostrar los modos de los operadores
            st.write("Selector mode:", config["selection"].mode)
            st.write("Crosser mode:", config["crossover"].mode)
            st.write("Mutator mode:", config["mutation"].mode)
            st.write("Replacer mode:", config["replacement"].mode)

            # Mostrar las medidas de convergencia si se habilita en la configuración
            if config["measure_convergence"]:
                st.subheader("Medidas de convergencia")
                st.write("Convergencia de selección:", config["selection"].measures['convergences'])
                st.write("Convergencia de cruce:", config["crossover"].measures['convergences'])
                st.write("Convergencia de mutación:", config["mutation"].measures['convergences'])
                st.write("Convergencia de reemplazo:", config["replacement"].measures['convergences'])

                # Mostrar gráficos de convergencia si se habilita
                convergences_data = [
                    config["selection"].measures['convergences'],
                    config["crossover"].measures['convergences'],
                    config["mutation"].measures['convergences'],
                    config["replacement"].measures['convergences'],
                ]
                labels = ["Selección", "Cruce", "Mutación", "Reemplazo"]

                try:
                    fig = plot.plot_convergences(np.array(convergences_data, dtype=object), labels)
                    st.subheader("Gráfica de convergencia de los operadores")
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error generando la gráfica de convergencias: {e}")


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


def deap_genetic_algorithm(config):
    # Crear el problema de minimización
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Definir toolbox
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.uniform, -1, 1)  # Rango inicial de genes
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=config["num_coef"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness function igual que tu 'utils.fitness'
    def evaluate(individual):
        return utils.fitness(np.array(individual)),

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Crear la población
    pop = toolbox.population(n=config["pop_size"])

    # Ejecutar algoritmo
    start = time.time()
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(pop, toolbox,
                        cxpb=0.5, mutpb=0.2,
                        ngen=config["generations"],
                        stats=stats, halloffame=hof,
                        verbose=False)
    end = time.time()

    best = hof[0]
    return {
        'coefficients': np.array(best),
        'error': utils.fitness(np.array(best)),
        'time': end - start
    }



def plot_comparison(result_own, result_deap):
    """
    Muestra un gráfico de barras comparando los resultados de ambos algoritmos.
    
    Args:
        result_own (dict): Resultados de tu algoritmo (ej. error, fitness).
        result_deap (dict): Resultados del algoritmo DEAP (ej. error, fitness).
    """
    # Suponemos que ambos resultados tienen un campo 'error' o 'fitness'
    error_own = result_own['error'] if 'error' in result_own else None
    error_deap = result_deap['error'] if 'error' in result_deap else None

    # Crear el gráfico de barras para la comparación
    fig = go.Figure(data=[
        go.Bar(name='Tu Algoritmo', x=['Error'], y=[error_own], marker=dict(color='blue')),
        go.Bar(name='DEAP Algoritmo', x=['Error'], y=[error_deap], marker=dict(color='orange'))
    ])

    # Configuración del gráfico
    fig.update_layout(
        title="Comparación de Errores entre Algoritmos",
        xaxis_title="Algoritmo",
        yaxis_title="Error",
        barmode='group',  # Agrupar las barras
        plot_bgcolor='white'
    )

    return fig

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


# Comparar resultados con el algoritmo DEAP
if "AG" in st.session_state:
    result_own = st.session_state['AG']
else:
    result_own = None  # Si no existe, se asigna None

result_deap = deap_genetic_algorithm(config)

st.subheader("Comparación de Resultados")

col1, col2 = st.columns(2)
with col1:
    st.write("🚀 Tu Algoritmo")
    if result_own:
        st.write(result_own)
    else:
        st.write("Aún no has ejecutado tu algoritmo.")
        
with col2:
    st.write("DEAP Algoritmo")
    st.write(result_deap)
