import algorithms.genetic as ag

import tools.utils as utils

import functions.crossing as cross
import functions.mutation as mutate
import functions.replacement as replace
import functions.selection as select

seltc = select.tournament(2, utils.fitness, True)
crossar = cross.arithmetic(utils.fitness, True)
mutga = mutate.gaussian(0.1,utils.fitness, True)
replac = replace.total(utils.fitness, True)


ag.genetic_function_optimization(
    utils.population(10, 5), 
    10, 
    100,
    seltc,
    crossar,
    mutga,
    replac,
    utils.fitness
)

# print("Convergencia de los selectores", seltc.convengences)
# print("Convergencia de los cruces", crossar.convengences)
# print("Convergencia de las mutaciones", mutga.convengences)
# print("Convergencia de los reemplazos", replac.convengences)

formato = {
    "id": "convergenciasXXXX",
    "selector": seltc.convengences,
    "cruce": crossar.convengences,
    "mutacion": mutga.convengences,
    "reemplazo": replac.convengences
}

SAVEPATH = "info/"

utils.save(formato, SAVEPATH + formato["id"] + ".json")

print(utils.best_algorithm_config())