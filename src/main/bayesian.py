from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from tools.utils import fitness, population
import algorithms.genetic as gnc
import functions.crossing as cross
import functions.mutation as mutate
import functions.replacement as replace
import functions.selection as select



# Para los selectores

def optimize_genetic_algorithm():
    """
    Optimiza los parámetros del algoritmo genético utilizando optimización bayesiana.
    
    Returns:
        dict: Mejores parámetros encontrados.
    """
    # Espacio de búsqueda para los hiperparámetros
    space = [
        # Integer(1, 2,name='num_islands'),  # Número de islas
        Integer(10, 200, name='pop_size'),  # Tamaño de la población
        Integer(10, 500, name='generations'),  # Número de generaciones
        Categorical(['Torneo Binario'], name='selection_method'),
        Categorical(['Cruce Aritmético', 'Cruce de Un Punto', 'Cruce Uniforme', 'Cruce BLX'], name='crossover_method'),
        Categorical(['Mutación Gaussiana'], name='mutation_method'),
        Categorical(['Reemplazo Generacional Completo'], name='replacement_method'),
        Real(0.01, 1.0, name='mutation_sigma'),  # Parámetro para mutación gaussiana
        Integer(2, 10, name='tournament_k')  # Parámetro para selección por torneo
    ]

    @use_named_args(space)
    def objective(**params):
        """
        Función objetivo para la optimización bayesiana.
        
        Args:
            params (dict): Parámetros actuales.
        
        Returns:
            float: Error cuadrático medio (MSE) del mejor individuo.
        """
        try:
            # Validar parámetros
            if params['mutation_sigma'] <= 0 or params['mutation_sigma'] > 1:
                return 1e10  # Penalizar valores inválidos
            if params['pop_size'] <= 0 or params['generations'] <= 0:
                return 1e10

            # Configurar los operadores según los parámetros
            selection = select.selections()[params['selection_method']]
            if params['selection_method'] == 'Torneo Binario':
                selection = selection(min(params['tournament_k'], params['pop_size']), fitness)
            else:
                selection = selection(fitness)

            crossover = cross.crossings()[params['crossover_method']](fitness)
            mutation = mutate.mutations()[params['mutation_method']]
            if params['mutation_method'] == 'Mutación Gaussiana':
                mutation = mutation(params['mutation_sigma'], fitness)
            else:
                mutation = mutation(fitness)

            replacement = replace.replacements()[params['replacement_method']](fitness)

            result = gnc.genetic_function_optimization(
                population(params['pop_size'], 8),
                params['pop_size'],
                params['generations'],
                selection,
                crossover,
                mutation,
                replacement,
                fitness
            )
            return result['error'] + params['generations'] * 0.001  # Minimizar el error cuadrático medio (MSE)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 1e10 # Penalizar errores

    # Ejecutar la optimización bayesiana
    res = gp_minimize(objective, space, n_calls=20,verbose=True)

    # Retornar los mejores parámetros encontrados
    best_params = {dim.name: val for dim, val in zip(space, res.x)}
    return best_params


if __name__ == "__main__":
    best_params = optimize_genetic_algorithm()
    print("Mejores parámetros encontrados:")
    for param, value in best_params.items():
        print(f"{param}: {value}")