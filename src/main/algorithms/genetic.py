import concurrent.futures as futures
import tools.utils as utils
import numpy as np
import gc

def diversity(island: np.array) -> float:
    """
    Calcula la diversidad de una isla como la media de las distancias entre todos los individuos.

    Args:
        island (np.array): Población de la isla (matriz de individuos).

    Returns:
        float: Diversidad de la isla.
    """
    try:
        _island = island[~np.isnan(island).any(axis=1)]
        distances = np.linalg.norm(_island[:, np.newaxis] - _island, axis=2)
        max_distance = np.nan_to_num(np.max(distances), nan=0.0)
        max_distance = np.max(distances)
        mean_distance = (np.mean(distances) + np.std(distances)) / max_distance if max_distance != 0 else 0.1
        # print(f"Normalized mean distance: {mean_distance}")
        return mean_distance
    except Exception as e:
        print(f"Error calculating diversity: {e}")
        return 0.1

def genetic_function_optimization(island: np.array, pop_size: int,
                                  generations: int, select: callable,
                                  cross: callable, mutate: callable,
                                  replace: callable, fitness: callable) -> dict:
    """
    Implementa la evolución genética para una única isla.

    Args:
        island (np.array): Población inicial de la isla (matriz de individuos).
        pop_size (int): Tamaño de la población en la isla.
        generations (int): Número de generaciones a ejecutar.
        select (callable): Función de selección que elige individuos para reproducirse.
        cross (callable): Función de cruce que genera descendencia a partir de dos padres.
        replace (callable): Función de reemplazo que actualiza la población con la nueva generación.
        fitness (callable): Función de fitness que evalúa la calidad de un individuo.

    Returns:
        dict: Diccionario con:
            - 'coefficients': El mejor individuo encontrado (array de coeficientes).
            - 'error': El valor de fitness del mejor individuo.

    Notas:
        - La función utiliza `np.empty` para crear una nueva población (`new_island`) en cada generación.
        - Libera explícitamente la memoria de las variables `island` y `new_island` al finalizar.
    """
    try:
        mutation_rate = diversity(island)

        for _ in range(generations):
            # Crear una nueva población vacía
            new_island = np.empty(island.shape, dtype=island.dtype)
            for i in range(pop_size // 2):
                # Selección de padres
                p1, p2 = select(island)

                # Cruce y mutación
                ch1, ch2 = cross(p1, p2), cross(p2, p1)
                ch1, ch2 = mutate(ch1, mutation_rate), mutate(ch2, mutation_rate)
                new_island[i*2], new_island[i*2+1] = ch1, ch2

            # Reemplazo de la población
            replace(island, new_island)
            # Actualizar la diversidad
            mutation_rate = diversity(island)
            # print(mutation_rate)

        # Encontrar el mejor individuo
        solution = min(island, key=fitness)
        return {'coefficients': solution, 'error': fitness(solution)}
    finally:
        # Liberar memoria explícitamente
        del island
        del new_island


def parallelize(evolver: callable, num_islands: int, pop_size: int, generations: int,
                num_coef: int, select: callable, cross: callable, mutate: callable,
                replace: callable, fitness: callable) -> dict:
    """
    Paraleliza la ejecución del modelo de islas utilizando `ProcessPoolExecutor`.

    Args:
        evolver (callable): Función que implementa la lógica de evolución para una isla.
        num_islands (int): Número de islas a crear.
        pop_size (int): Tamaño de la población en cada isla.
        generations (int): Número de generaciones a ejecutar en cada isla.
        num_coef (int): Número de coeficientes (genes) por individuo.
        select (callable): Función de selección que elige individuos para reproducirse.
        cross (callable): Función de cruce que genera descendencia a partir de dos padres.
        mutate (callable): Función de mutación que modifica un individuo.
        replace (callable): Función de reemplazo que actualiza la población con la nueva generación.
        fitness (callable): Función de fitness que evalúa la calidad de un individuo.

    Returns:
        dict: Diccionario con:
            - 'coefficients': El mejor individuo encontrado entre todas las islas.
            - 'error': El valor de fitness del mejor individuo.

    Notas:
        - Utiliza `ProcessPoolExecutor` para ejecutar la evolución de cada isla en paralelo.
        - Maneja excepciones en los procesos paralelos e imprime errores si ocurren.
        - Libera memoria explícitamente utilizando `gc.collect()` después de procesar cada futuro.
    """
    solution = None
    with futures.ProcessPoolExecutor(max_workers=num_islands) as executor:
        # Crear una lista de futuros para ejecutar la evolución en paralelo
        futures_list = [
            executor.submit(evolver, utils.population(pop_size, num_coef), pop_size, generations,
                            select, cross, mutate, replace, fitness) for _ in range(num_islands)
        ]
        for future in futures.as_completed(futures_list):
            try:
                # Obtener el resultado del futuro
                result = future.result()
                if solution is None or result['error'] < solution['error']:
                    solution = result
            except Exception as e:
                print(f"Error: {e}")
            finally:
                # Forzar la recolección de basura
                gc.collect()

    return list(solution)


@utils.measure
def island_optimization(num_islands: int, pop_size: int, generations: int, num_coef: int,
                        select: callable, cross: callable, mutate: callable, # mutation_rate: float,
                        replace: callable, fitness: callable) -> dict:
    """
    Ejecuta el modelo de islas y mide el tiempo y el uso de memoria.

    Args:
        num_islands (int): Número de islas a crear.
        pop_size (int): Tamaño de la población en cada isla.
        generations (int): Número de generaciones a ejecutar en cada isla.
        num_coef (int): Número de coeficientes (genes) por individuo.
        select (callable): Función de selección que elige individuos para reproducirse.
        cross (callable): Función de cruce que genera descendencia a partir de dos padres.
        mutate (callable): Función de mutación que modifica un individuo.
        replace (callable): Función de reemplazo que actualiza la población con la nueva generación.
        fitness (callable): Función de fitness que evalúa la calidad de un individuo.

    Returns:
        dict: Diccionario con:
            - 'coefficients': El mejor individuo encontrado entre todas las islas.
            - 'error': El valor de fitness del mejor individuo.
            - 'time': Tiempo total de ejecución.
            - 'memory': Uso máximo de memoria durante la ejecución.

    Notas:
        - Utiliza el decorador `@utils.measure` para medir el tiempo y el uso de memoria.
        - Es la interfaz principal para ejecutar el modelo de islas.
    """
    return parallelize(genetic_function_optimization, num_islands, pop_size,
                       generations, num_coef, select, cross, mutate, replace, fitness)