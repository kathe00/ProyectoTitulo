import time
import numpy as np
from FeatureSelection.Instancia import Instancia 
from FeatureSelection.FSproblem import FeatureSelection
from Metaheuristicas.MFO import MothFlame
from Metaheuristicas.AVOA import AfricanVultures
from Metaheuristicas.Binarizacion import binarizacion
"""
SOLVER FEATURE SELECTION - Módulo para la resolución de Feature Selection Problem
 Input:
    - instancia: nombre de la instancia
    - pop: tamaño de la población
 Output:
    -
"""

def solverFS(instancia, pop, k, gamma, maxIter, esquema, ub, lb, mh):
    # INICIALIZACIÓN
    print("-----------------------------------------------------------------")
    print("Resolviendo la instancia ["+ instancia + "] en Feature Selection")
    print("-----------------------------------------------------------------")
    print(" \n- INICIALIZACIÓN -")
    print("Tamaño de población: " + str(pop))
    # - leer la instancia
    ins = Instancia()
    ins.leerInstancia(instancia)
    datos = ins.getDatos()
    clases = ins.getClases()
    print("Instancia leída.")

    # - instanciar límites
    dim = len(datos.columns)
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # - inicializar FS
    fs = FeatureSelection(datos, clases, len(datos.columns), gamma)
    fitness = np.zeros(pop)       # fitness
    accuracy = np.zeros(pop)      # presición pred. correctas
    f1Score = np.zeros(pop)       # puntuación F1
    presicion = np.zeros(pop)     # presición pred. positivas
    recall = np.zeros(pop)        # exhaustividad
    mcc = np.zeros(pop)           # coeficiente de correlación Matthews
    errorRate = np.zeros(pop)     # rango de error
    totalSelected = np.zeros(pop) # cantidad de carac. seleccionadas

    # - primera población (todas las características)
    poblacion = np.ones((pop, dim))
    print("Primera población generada.")

    # - factibilidad y fitness
    for i in range(pop):
        while not ins.esFactible(poblacion[i]): # mientras sea infactible
            poblacion[i] = fs.nuevaSolucion(1)  # genero una nueva
        
        # calcular fitness
        fitness[i], accuracy[i], f1Score[i], presicion[i], recall[i], mcc[i], errorRate[i], totalSelected[i] = fs.fitness(poblacion[i],k)
    
    print("Fitness inicial calculado, valores:")
    print(fitness)

    # - definir mejor solución y fitness
    orden = fitness.argsort(axis=0)
    mejorIdx = orden [0] # index del mejor fitness

    fitnessRanking = np.zeros(pop)
    fitnessRanking = fitness[orden] # fitness ordenados de mejor a peor
    bestFitness = fitnessRanking[0] # mejor fitness
    print("Mejor fit: " + str(bestFitness))

    solutionsRanking = np.zeros(pop)
    solutionsRanking = poblacion[orden, :]    # soluciones ordenadas de mejor a peor
    bestSolution = solutionsRanking[0].copy() # mejor solución
    print("Mejor solución: " + str(bestSolution))

    # medidas de la mejor solución
    bestAccuracy = accuracy[mejorIdx]
    bestF1Score = f1Score[mejorIdx]
    bestPresicion = presicion[mejorIdx]
    bestRecall = recall[mejorIdx]
    bestMcc = mcc[mejorIdx]
    bestErrorRate = errorRate[mejorIdx]
    bestTFS = totalSelected[mejorIdx]

    # INICIO DE LA OPTIMIZACIÓN ----------------------------------------------------------------------------------
    print(" \n- OPTIMIZACIÓN -")
    time1 = time.time()
    binSol = poblacion.copy() # la primera solución es binaria

    # - Moth Flame Optimization
    # instanciar metaheurística
    mfo = MothFlame(pop, dim, maxIter, ub, lb)
    avoa = AfricanVultures(pop, dim, ub, lb, maxIter, 0.5,0.5,0.5,2,3)

    # Iteración principal
    for iteration in range(maxIter):

        # Segunda y tercera iteración (perturvar población)
        if( mh == 'MFO' ):
            poblacion, solutionsRanking, fitnessRanking = mfo.iterar(iteration, poblacion, fitness, solutionsRanking, fitnessRanking)
        if( mh == 'AVOA' ):
            poblacion = avoa.iterar(poblacion, fitness, solutionsRanking)
        
        # Procesamiento FS
        for i in range(pop):
            # binarizar
            poblacion[i] = binarizacion.aplicarBinarizacion(poblacion[i], esquema[0], esquema[1], bestSolution, binSol[i].tolist())

            # comprobar factibilidad
            while not ins.esFactible(poblacion[i]): # mientras sea infactible
                print("** Infactible! ** ")
                print(poblacion[i])
                poblacion[i] = fs.nuevaSolucion(1)  # genero una nueva
            
            # calcular fitness
            fitness[i], accuracy[i], f1Score[i], presicion[i], recall[i], mcc[i], errorRate[i], totalSelected[i] = fs.fitness(poblacion[i],k)

            # verificar mejor fitness
            if fitness[i] > bestFitness:
                bestAccuracy = accuracy[i]
                bestF1Score = f1Score[i]
                bestPresicion = presicion[i]
                bestRecall = recall[i]
                bestMcc = mcc[i]
                bestErrorRate = errorRate[i]
                bestTFS = totalSelected[i]
        
        # copiar la poblacion binarizada
        binSol = poblacion.copy()

        # obtener rankings
        solutionsRanking, fitnessRanking = mfo.ordenarFlamas(solutionsRanking, poblacion, fitnessRanking, fitness)
        bestFitness = fitnessRanking[0]
        bestSolution = solutionsRanking[0].copy()
        print("Iteracion " + str(iteration))
        print(" - Best Fitness: " + str(bestFitness))

    print(" \n- FIN OPTIMIZACIÓN -")
    print(" Best Fitness: " + str(bestFitness)
          + "\nBest Solution: " + str(bestSolution)
          + "\nAccuracy: " + str(bestAccuracy)
          + "\nF1 Score: " + str(bestF1Score)
          + "\nPresicion: " + str(bestPresicion)
          + "\nRecall: " + str(bestRecall)
          + "\nMCC: " + str(bestMcc)
          + "\nError rate: " + str(bestErrorRate)
          + "\nFeatures Selected: " + str(bestTFS))

    time2 = time.time()
    tiempoEjec = time2 - time1
    print("Tiempo de ejecución: " + str(tiempoEjec))