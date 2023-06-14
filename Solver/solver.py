"""
SOLVER GENERAL 
Módulo para la resolución de los problemas implementados en el proyecto
"""

from Solver.solverFS import solverFS
from Solver.solverBench import solverBench
from Solver.solverSCP import solverSCP
import matplotlib.pyplot as plt

# ------------------------------ FEATURE SELECTION ------------------------------ #
def FeatureSelection(instancia, datosMH, paramMH, paramProblem):
    # - EJECUCIÓN -
    bestFit, bestSol, performance, curvaConvergencia = solverFS(instancia, datosMH, paramMH, paramProblem)
    #
    plt.plot(range(len(curvaConvergencia)), curvaConvergencia)
    plt.show()

    # - RESULTADOS -
    if (bestFit != -1):
        print("\n-----------------------------------------------------------------")
        print("-------------------------- Resultados ---------------------------")
        print("Best Fitness: " + str(bestFit))
        print(bestSol)
        print("Desempeño del clasificador:")
        print("Accuracy: " + str(performance[0])
            + "\nF1 Score: " + str(performance[1])
            + "\nPrecision: " + str(performance[2])
            + "\nRecall: " + str(performance[3])
            + "\nMCC: " + str(performance[4])
            + "\nError Rate: " + str(performance[5])
            + "\nCant. Features Selected: " + str(performance[6])
            + "\nTiempo de ejecución: " + str(performance[7]))
        
def BenchMark(instancia, datosMH, paramMH, paramProblem):
    # - EJECUCIÓN -
    bestFit, bestSol, performance, curvaConvergencia = solverBench(instancia, datosMH, paramMH, paramProblem)
    #
    plt.plot(range(len(curvaConvergencia)), curvaConvergencia)
    plt.show()
    # - RESULTADOS -
    if (bestFit != -1):
        print("\n-----------------------------------------------------------------")
        print("-------------------------- Resultados ---------------------------")
        print("Best Fitness: " + str(bestFit))
        print(bestSol)
        print("Tiempo de ejecución: " + str(performance[0]))

def SetCoveringProblem(instancia, datosMH, paramMH, paramProblem):
    # - EJECUCIÓN -
    bestFit, bestSol, performance, curvaConvergencia = solverSCP(instancia, datosMH, paramMH, paramProblem)
    #
    plt.plot(range(len(curvaConvergencia)), curvaConvergencia)
    plt.show()
    # - RESULTADOS -
    if (bestFit != -1):
        print("\n-----------------------------------------------------------------")
        print("-------------------------- Resultados ---------------------------")
        print("Best Fitness: " + str(bestFit))
        print(bestSol)
        print("Tiempo de ejecución: " + str(performance[0]))