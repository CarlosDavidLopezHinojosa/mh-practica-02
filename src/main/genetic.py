import multiprocessing as mp
import random as rnd
import math 
from utils import measure

def migrate(islands, island, island_id, fitness, lock):
    """
    Realiza la migración de individuos entre islas.
    """
    with lock:
        # Obtener el mejor individuo de la isla actual
        if len(island) == 0:
            return  # Si la isla actual está vacía, no se puede migrar

        my_apex = min(list(island), key=fitness)

        # Seleccionar otra isla aleatoria
        other_island = rnd.choice([i for i in range(len(islands)) if i != island_id])

        # Verificar que la otra isla no esté vacía
        if len(islands[other_island]) == 0:
            return  # Si la otra isla está vacía, no se puede migrar

        # Obtener el mejor individuo de la otra isla
        other_apex = min(islands[other_island], key=fitness)

        # Realizar el intercambio de individuos
        island[rnd.randint(0, len(island) - 1)] = other_apex
        islands[other_island][rnd.randint(0, len(islands[other_island]) - 1)] = my_apex

def genetic_function_optimization(islands, island, island_id, pop_size, generations, select, cross, mutate, replace, fitness, lock, results):

    migration_ratio = int(rnd.uniform(math.log(generations), generations ** 0.5))
    for generation in range(generations):
        new_island = []
        for _ in range(pop_size // 2):
            
            p1, p2 = select(island, fitness), select(island, fitness)
            while p1 is p2: # Evita hijos de padres iguales
                p2 = select(island, fitness)
            
            ch1, ch2 = cross(p1, p2), cross(p2, p1)
            ch1, ch2 = mutate(ch1), mutate(ch2)

            new_island.extend([ch1, ch2])

        replace(island, new_island)
        if generation % migration_ratio == 0:
            migrate(islands, island, island_id, fitness, lock)

    solution = min(island, key=fitness)
    results.append({'coefficients': solution, 'error': fitness(solution)})      

def manager_islands(num_islands, pop_size, num_coef, mng):
    
    return mng.list([
        mng.list([
            mng.list([rnd.random() for _ in range(num_coef)]) 
            for _ in range(pop_size)
        ]) 
        for island in range(num_islands)
    ])


def parallelize(evolver, num_islands, pop_size, generations, num_coef, select, cross, mutate, replace, fitness):

    mng = mp.Manager()
    lck = mp.Lock()
    results = mng.list()

    islands = manager_islands(num_islands, pop_size, num_coef, mng)
    
    processes = []
    for i in range(num_islands):
        p = mp.Process(target=evolver, args=(islands, islands[i], i, pop_size, generations, select, cross, mutate, replace, fitness, lck, results))
        processes.append(p)
        p.start()

    try:
        for p in processes:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        print("Execution interrupted. Terminating processes...")
        for p in processes:
            p.terminate()
            p.join()

    return min(list(results), key=lambda x : x['error'])

@measure
def island_optimization(num_islands, pop_size, generations, num_coef, select, cross, mutate, replace, fitness):
    return parallelize(genetic_function_optimization, num_islands, pop_size, generations, num_coef, select, cross, mutate, replace, fitness)