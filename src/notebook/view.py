import streamlit as st
import operators as ops
import parallel as prl
import utils


# Título de la aplicación
st.title("Optimización Genética con Modelo de Islas")

# Descripción del proyecto
st.markdown("""
Esta aplicación utiliza un algoritmo genético basado en el modelo de islas para resolver problemas de optimización.
El modelo divide la población en subpoblaciones (islas) que evolucionan de forma independiente y ocasionalmente migran individuos entre ellas.
""")

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
def selection_tournament(population, fitness, k=2):
    tournament = random.sample(list(population), k)
    return min(tournament, key=fitness)
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
def crossing_arithmetic(parent1, parent2):
    alpha = random.random()
    return [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
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
def mutation_gaussian(individual, mutation_rate=0.1, sigma=0.1):
    return [
        gene + random.gauss(0, sigma) if random.random() < mutation_rate else gene
        for gene in individual
    ]
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
def replacement_elitism(population, new_population):
    population.sort(key=lambda x: x[1])
    new_population.sort(key=lambda x: x[1])
    for i in range(len(new_population)):
        population[i] = new_population[i]
""", language="python")

# Sección: Algoritmo Genético (Optimización)
st.header("Algoritmo Genético (Optimización)")
st.markdown("""
El algoritmo genético basado en el modelo de islas divide la población en subpoblaciones (islas) que evolucionan de forma independiente. 
Cada cierto número de generaciones, los mejores individuos migran entre islas para mantener la diversidad genética.
""")
st.code("""
def genetic_function_optimization(islands: list[list], island_id: int, 
                                  pop_size: int, generations: int, 
                                  select: callable, cross: callable, 
                                  mutate: callable, replace:callable, 
                                  fitness:callable, lock: object, 
                                  results: object) -> tuple:
    manager_ratio = utils.migration_ratio(generations)
    island = islands[island_id]

    for generation in range(generations):
        new_island = []
        for _ in range(pop_size // 2):
            p1, p2 = select(island, fitness), select(island, fitness)
            while p1 is p2:
                p2 = select(island, fitness)

            ch1, ch2 = cross(p1, p2), cross(p2, p1)
            ch1, ch2 = mutate(ch1), mutate(ch2)
            new_island.extend([ch1, ch2])

        replace(island, new_island)
        if generation % manager_ratio == 0:
            prl.migrate(islands, island, island_id, fitness, lock)

    solution = min(island, key=fitness)
    with lock:
        results.append({'coefficients': solution, 'error': fitness(solution)})
""", language="python")

# Parámetros de entrada
st.sidebar.header("Parámetros del Algoritmo")
num_islands = st.sidebar.number_input("Número de Islas", min_value=1, max_value=20, value=5, step=1)
pop_size = st.sidebar.number_input("Tamaño de la Población por Isla", min_value=1, max_value=100, value=20, step=1)
generations = st.sidebar.number_input("Número de Generaciones", min_value=1, max_value=1000, value=100, step=10)
num_coef = 8

# Botón para ejecutar el algoritmo
if st.button("Ejecutar Algoritmo"):
    with st.spinner("Ejecutando el modelo de islas..."):

        if __name__ == '__main__':
            result = prl.island_optimization(
            num_islands, pop_size, generations, num_coef,
            ops.selection_tournament, 
            ops.crossing_arithmetic, 
            ops.mutation_gaussian, 
            ops.replacement_elitism, 
            utils.fitness
        )
        st.success("Optimización completada.")
        st.subheader("Resultado")
        st.write(result)