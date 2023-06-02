"""
PROYECTO DE TÍTULO - INGENIERÍA DE EJECUCIÓN INFORMÁTICA 2023
Katherine Sepúlveda
"""

from Solver.solverFS import solverFS
from Solver.solverBench import solverBench
from Metaheuristicas.AVOA import AfricanVultures

# - PARÁMETROS - #
instancia = 'ionosphere'

datosMH = 'MFO,20,-2,2'
paramMH = ''
paramFS = '100,3,0.99,S2,ELIT'

bestFitMFO, bestSolMFO = solverFS(instancia, datosMH, paramMH, paramFS)

datosMH = 'AVOA,20,-2,2'
paramMH = '0.5,0.5,0.5,0.5,3'
paramFS = '100,3,0.99,S2,ELIT'

bestFitAVOA, bestSolAVOA = solverFS(instancia, datosMH, paramMH, paramFS)

print("--------------------")
print("-- Resultados MFO --")
print("Best Fitness: " + str(bestFitMFO))
print(bestSolMFO)
print("---------------------")
print("-- Resultados AVOA --")
print("Best Fitness: " + str(bestFitAVOA))
print(bestSolAVOA)

# ub = 1
# lb = -1
# dim = 30
# solverBench(tamPop, dim, maxIter, ub, lb, 'F3','AVOA')

