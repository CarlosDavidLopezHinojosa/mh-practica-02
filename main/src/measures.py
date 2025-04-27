import algorithms.genetic as ag

import tools.utils as utils

import functions.crossing as cross
import functions.mutation as mutate
import functions.replacement as replace
import functions.selection as select


SAVEPATH = "main/info/"
BESTALGCONFIG = utils.best_algorithm_config(mode=True)

def measure_selectors():
    results = {}
    for name, selector in select.selections().items():
            
        config = BESTALGCONFIG.copy()
        if name == "Aleatorio" or name == "Ruleta" or name == "Emparejamiento Variado Inverso":
            config['select'] = selector(7, config["fitness"], True)

        print("Midiendo:", name)
        _ = ag.genetic_function_optimization(utils.population(config["pop_size"], 8),
            config["pop_size"], config["generations"], config["select"],
            config["cross"], config["mutate"], config["replace"], config["fitness"]
        )
        
        
        results[name] = config["select"].measures
    utils.save(results, SAVEPATH + "selectors.json")

def measure_crossings():
    results = {}
    for name, crossing in cross.crossings().items():
        config = BESTALGCONFIG.copy()
        if name != "BLX":
            config['cross'] = crossing(config["fitness"], True)
        print("Midiendo:", name)
        _ = ag.genetic_function_optimization(utils.population(config["pop_size"], 8),
            config["pop_size"], config["generations"], config["select"],
            config["cross"], config["mutate"], config["replace"], config["fitness"]
        )
        
        
        results[name] = config["cross"].measures
    utils.save(results, SAVEPATH + "crossings.json")

def measure_mutations():
    results = {}
    for name, mutation in mutate.mutations().items():
        config = BESTALGCONFIG.copy()
        if name != "Mutación Gaussiana":
            config['mutate'] = mutation(utils.fitness, True)

        if name == "Mutación No Uniforme":
            config['mutate'] = mutation(config["generations"],config["fitness"], True)
            
        print("Midiendo:", name)
        _ = ag.genetic_function_optimization(utils.population(config["pop_size"], 8),
            config["pop_size"], config["generations"], config["select"],
            config["cross"], config["mutate"], config["replace"], config["fitness"]
        )
        
        
        results[name] = config["mutate"].measures
    utils.save(results, SAVEPATH + "mutations.json")

def measure_replacements():
    results = {}
    for name, replacement in replace.replacements().items():
        config = BESTALGCONFIG.copy()
        if name == "Torneo restringido" or name == "Peor entre semejantes":
            config['replace'] = replacement(7, config["fitness"], True)
        elif name != "Elitismo":
            config['replace'] = replacement(config["fitness"], True)
        print("Midiendo:", name)
        _ = ag.genetic_function_optimization(utils.population(config["pop_size"], 8),
            config["pop_size"], config["generations"], config["select"],
            config["cross"], config["mutate"], config["replace"], config["fitness"]
        )
        
        
        results[name] = config["replace"].measures
    utils.save(results, SAVEPATH + "replacements.json")

if __name__ == "__main__":
    # measure_selectors()
    # measure_crossings()
    # measure_mutations()
    # measure_replacements()
    pass