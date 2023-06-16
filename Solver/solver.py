"""
SOLVER GENERAL 
Módulo para la resolución de los problemas implementados en el proyecto
"""

from Solver.solverFS import solverFS
from Solver.solverBench import solverBench
from Solver.solverSCP import solverSCP
import matplotlib.pyplot as plt

# ------------------------------ FEATURE SELECTION ------------------------------ #
def FeatureSelection(instancia, datosMH, paramMH, paramProblem, fecha):
    # - EJECUCIÓN -
    bestFit, bestSol, performance, curvaConvergencia, graficoExpl, graficoPerf = solverFS(instancia, datosMH, paramMH, paramProblem)

    # - RESULTADOS -
    if (bestFit != -1):

        # directorio
        directorio = "./Resultados/" + fecha + "/"

        # añadimos a la descripcion (fecha) la mh utilizada
        aux = fecha.split("_",1)
        fecha = aux[0] + "-" + datosMH.split(',')[0] + "_" + aux[1]

        # - Gráficos
        # curva convergencia
        plt.figure()
        plt.title("Curva de Convergencia")
        plt.plot(range(len(curvaConvergencia)), curvaConvergencia)
        plt.savefig(directorio + fecha + '_CurvaConvergencia.png')

        # exploración y explotación
        plt.figure()
        plt.title("Explotación v/s Exploración")
        xpl = graficoExpl[:, 0]
        xpt = graficoExpl[:, 1]

        fig, ax = plt.subplots()

        ax.plot(xpl, label='Exploración')
        ax.plot(xpt, label='Explotación')

        ax.legend()
        plt.savefig(directorio + fecha + '_ExplorExplot.png')

        # performance
        plt.figure()
        plt.title("Accuracy")
        plt.plot(range(len(graficoPerf[0,:])), graficoPerf[0,:]) # Accuracy
        plt.savefig(directorio + fecha + '_Accuracy.png')

        plt.figure()
        plt.title("F1 Score")
        plt.plot(range(len(graficoPerf[0,:])), graficoPerf[1,:]) # F1score
        plt.savefig(directorio + fecha + '_F1Score.png')

        plt.figure()
        plt.title("Presicion")
        plt.plot(range(len(graficoPerf[0,:])), graficoPerf[2,:]) # Presicion
        plt.savefig(directorio + fecha + '_Presicion.png')

        plt.figure()
        plt.title("Recall")
        plt.plot(range(len(graficoPerf[0,:])), graficoPerf[3,:]) # Recall
        plt.savefig(directorio + fecha + '_Recall.png')

        plt.figure()
        plt.title("MCC")
        plt.plot(range(len(graficoPerf[0,:])), graficoPerf[4,:]) # MCC
        plt.savefig(directorio + fecha + '_MCC.png')

        plt.figure()
        plt.title("Error Rate")
        plt.plot(range(len(graficoPerf[0,:])), graficoPerf[5,:]) # ErrorRate
        plt.savefig(directorio + fecha + '_ErrorRate.png')

        plt.figure("Features Selected")
        plt.plot(range(len(graficoPerf[0,:])), graficoPerf[6,:]) #FeatureSelected
        plt.savefig(directorio + fecha + '_FeatureSelected.png')

        # Archivo de resultados
        archivo = open(directorio + "resultados.txt", "a", encoding="utf-8")
        archivo.write("--------------------------------------------------------------------")
        archivo.write("\nRESULTADOS DE LA EJECUCIÓN\n")
        archivo.write("\nProblema: Feature Selection \nInstancia: " + instancia)
        archivo.write("\nMetaheurística: " + datosMH.split(',')[0])
        archivo.write("\nBest Fitness: " + str(bestFit))
        archivo.write("\nMejor solución:\n")
        for feature in bestSol:
            archivo.write(str(int(feature)) + " ")
        archivo.write("\n\n-- Desempeño del clasificador:")
        archivo.write("\nAccuracy: " + str(performance[0])
            + "\nF1 Score: " + str(performance[1])
            + "\nPrecision: " + str(performance[2])
            + "\nRecall: " + str(performance[3])
            + "\nMCC: " + str(performance[4])
            + "\nError Rate: " + str(performance[5])
            + "\nCant. Features Selected: " + str(performance[6])
            + "\nTiempo de ejecución: " + str(performance[7]))
        archivo.write("\n\n")
        archivo.close()

        print("\n-------------------------- Resultados ---------------------------"
              + "\nBest Fitness: " + str(bestFit)
              + "\nMejor solución: " + str(bestSol)
              + "\nAccuracy: " + str(performance[0])
              + "\nTiempo de ejecución: " + str(performance[7])
              + "\n\nPara información más detallada diríjase al siguiente directorio:\n"
              + directorio[2:] + "resultados.txt")

# ------------------------------ BenchMarck ------------------------------ #
def BenchMark(instancia, datosMH, paramMH, paramProblem, fecha):
    # - EJECUCIÓN -
    bestFit, bestSol, performance, curvaConvergencia, graficoExpl = solverBench(instancia, datosMH, paramMH, paramProblem)

    
    # - RESULTADOS -
    if (bestFit != -1):

        # directorio
        directorio = "./Resultados/" + fecha + "/"

        # añadimos a la descripcion (fecha) la mh utilizada
        aux = fecha.split("_",1)
        fecha = aux[0] + "-" + datosMH.split(',')[0] + "_" + aux[1]
        
        # curva convergencia
        plt.title("Curva de Convergencia")
        plt.plot(range(len(curvaConvergencia)), curvaConvergencia)
        plt.savefig(directorio + fecha + '_CurvaConvergencia.png')

        # exploración y explotación
        plt.title("Explotación v/s Exploración")
        xpl = graficoExpl[:, 0]
        xpt = graficoExpl[:, 1]

        fig, ax = plt.subplots()

        ax.plot(xpl, label='Exploración')
        ax.plot(xpt, label='Explotación')

        ax.legend()
        plt.savefig(directorio + fecha + '_ExplorExplot.png')
    
        # Archivo de resultados
        archivo = open(directorio + "resultados.txt", "a", encoding="utf-8")
        archivo.write("--------------------------------------------------------------------")
        archivo.write("\nRESULTADOS DE LA EJECUCIÓN")
        archivo.write("\nProblema: Feature Selection \nInstancia: " + instancia)
        archivo.write("\nMetaheurística: " + datosMH.split(',')[0])
        archivo.write("\nBest Fitness: " + str(bestFit))
        archivo.write("\nMejor solución:\n")
        for feature in bestSol:
            archivo.write(str(feature) + " ")
        archivo.write("\nTiempo de ejecución: " + str(performance[0]))
        archivo.write("\n\n")
        archivo.close()

        # Mostrar por pantalla
        print("\n-------------------------- Resultados ---------------------------")
        print("Best Fitness: " + str(bestFit))
        print(bestSol)
        print("Tiempo de ejecución: " + str(performance[0]))
        print("\n\nPara información más detallada diríjase al siguiente directorio:\n"
              + directorio[2:] + "resultados.txt")

# ------------------------------ SetCovering ------------------------------ #
def SetCoveringProblem(instancia, datosMH, paramMH, paramProblem, fecha):
    # - EJECUCIÓN -
    bestFit, bestSol, performance, curvaConvergencia, graficoExpl = solverSCP(instancia, datosMH, paramMH, paramProblem)

    # - RESULTADOS -
    if (bestFit != -1):

        # directorio
        directorio = "./Resultados/" + fecha + "/"

        # añadimos a la descripcion (fecha) la mh utilizada
        aux = fecha.split("_",1)
        fecha = aux[0] + "-" + datosMH.split(',')[0] + "_" + aux[1]

        # curva convergencia
        plt.title("Curva de convergencia")
        plt.plot(range(len(curvaConvergencia)), curvaConvergencia)
        plt.savefig(directorio + fecha + '_CurvaConvergencia.png')

        # exploración y explotación
        plt.title("Explotación v/s Exploración")
        xpl = graficoExpl[:, 0]
        xpt = graficoExpl[:, 1]

        fig, ax = plt.subplots()

        ax.plot(xpl, label='Exploración')
        ax.plot(xpt, label='Explotación')

        ax.legend()
        plt.savefig(directorio + fecha + '_ExplorExplot.png')

        # Archivo de resultados
        archivo = open(directorio + "resultados.txt", "a", encoding="utf-8")
        archivo.write("--------------------------------------------------------------------")
        archivo.write("\nRESULTADOS DE LA EJECUCIÓN")
        archivo.write("\nProblema: Feature Selection \nInstancia: " + instancia)
        archivo.write("\nMetaheurística: " + datosMH.split(',')[0])
        archivo.write("\nBest Fitness: " + str(bestFit))
        archivo.write("\nMejor solución:\n")
        for feature in bestSol:
            archivo.write(str(int(feature)) + " ")
        archivo.write("\nTiempo de ejecución: " + str(performance[0]))
        archivo.write("\n\n")
        archivo.close()
        
        # Mostrar por pantalla
        print("\n-------------------------- Resultados ---------------------------")
        print("Best Fitness: " + str(bestFit))
        print(bestSol)
        print("Tiempo de ejecución: " + str(performance[0]))
        print("\n\nPara información más detallada diríjase al siguiente directorio:\n"
              + directorio[2:] + "resultados.txt")