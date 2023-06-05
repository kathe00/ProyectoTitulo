"""
PROYECTO DE TÍTULO - INGENIERÍA DE EJECUCIÓN INFORMÁTICA 2023
Katherine Sepúlveda
"""

from Solver.solverFS import solverFS
#from Solver.solverBench import solverBench
#from Metaheuristicas.AVOA import AfricanVultures

# - PARÁMETROS - #
instancia = 'ionosphere'

datosMH = 'MFO,20,-2,2'
paramMH = ''
paramFS = '10,3,0.99,S2,ELIT'

bestFitMFO, bestSolMFO, performanceMFO = solverFS(instancia, datosMH, paramMH, paramFS)

datosMH = 'AVOA,20,-2,2'
paramMH = '0.5,0.5,0.5,0.5,3'
paramFS = '10,3,0.99,S2,ELIT'

bestFitAVOA, bestSolAVOA, performanceAVOA = solverFS(instancia, datosMH, paramMH, paramFS)

if (bestFitMFO != -1):
    print("-----------------------------------------------------------------")
    print("----------------------- Resultados MFO --------------------------")
    print("Best Fitness: " + str(bestFitMFO))
    print(bestSolMFO)
    print("Desempeño del clasificador:")
    print("Accuracy: " + str(performanceMFO[0])
          + "\nF1 Score: " + str(performanceMFO[1])
          + "\nPrecision: " + str(performanceMFO[2])
          + "\nRecall" + str(performanceMFO[3])
          + "\nMCC: " + str(performanceMFO[4])
          + "\nError Rate: " + str(performanceMFO[5])
          + "\nCant. Features Selected: " + str(performanceMFO[6]))
    
if (bestFitAVOA != -1):
    print("-----------------------------------------------------------------")
    print("----------------------- Resultados AVOA -------------------------")
    print("Best Fitness: " + str(bestFitAVOA))
    print(bestSolAVOA)
    print("Desempeño del clasificador:")
    print("Accuracy: " + str(performanceAVOA[0])
          + "\nF1 Score: " + str(performanceAVOA[1])
          + "\nPrecision: " + str(performanceAVOA[2])
          + "\nRecall" + str(performanceAVOA[3])
          + "\nMCC: " + str(performanceAVOA[4])
          + "\nError Rate: " + str(performanceAVOA[5])
          + "\nCant. Features Selected: " + str(performanceAVOA[6]))

# ub = 1
# lb = -1
# dim = 30
# solverBench(tamPop, dim, maxIter, ub, lb, 'F3','AVOA')

