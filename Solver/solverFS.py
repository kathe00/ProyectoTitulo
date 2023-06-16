"""
SOLVER FEATURE SELECTION
Módulo para la resolución de Feature Selection Problem

Input: 
    - instancia: nombre de la instancia a resolver
    - datosMH: parámetros generales para la metaheurística ([nombre],[tamaño poblacion],[límite inferior],[límite superior])
    - paramMH: parámetros específicos de la metaheurística
                Si 'MFO': ([])
                Si 'AVOA': ([rp1],[rp2],[rp3],[l],[w])
    - paramFS: parámetros específicos de Feature Selection ([maxIteraciones],[k],[gamma],[función transf.],[tipo binarizacion])
Output:
    - bestFitness: mejor fitness encontrado
    - bestSolution: solución que obtuvo el mejor fitness
"""
import time
import numpy as np
from FeatureSelection.Instancia import Instancia 
from FeatureSelection.FSproblem import FeatureSelection
from Metaheuristicas.MFO import MothFlame
from Metaheuristicas.AVOA import AfricanVultures
from Metaheuristicas.Binarizacion import binarizacion
from Solver.diversidad import porcentajesXPLXPT, diversidadHussain


def solverFS(instancia, datosMH, paramMH, paramFS):
    # TOMAR DATOS
    mh = datosMH.split(',')[0]           # nombre de la metaheurística
    pop = int(datosMH.split(',')[1])     # tamaño de la población
    lb = int(datosMH.split(',')[2])      # límite inferior
    ub = int(datosMH.split(',')[3])      # límite superior

    maxIter = int(paramFS.split(',')[0]) # cantidad máxima de iteraciones
    k = int(paramFS.split(',')[1])       # cantidad de vecinos para KNN
    gamma = float(paramFS.split(',')[2]) # define alpha y beta de la función objetivo
    funcTransf = paramFS.split(',')[3]   # función de transferencia
    funcBin = paramFS.split(',')[4]      # función de binarización

    # INICIALIZACIÓN
    print("\n-----------------------------------------------------------------")
    print("Resolviendo la instancia ["+ instancia + "] en Feature Selection")
    print("-----------------------------------------------------------------")
    print(" \n- INICIALIZACIÓN -")
    print("Metaheurística: " + mh
          + "\nPoblación: " + str(pop)
          + "\nIteraciones: " + str(maxIter))
    
    # - leer la instancia
    ins = Instancia()
    ins.leerInstancia(instancia)
    datos = ins.getDatos()
    clases = ins.getClases()
    print("Instancia leída.")

    print(datos)

    # - instanciar límites (mismo valor)
    dim = len(datos.columns)      # dimensión del problema
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # - inicializar FS
    fs = FeatureSelection(datos, clases, dim, gamma, k)
    fitness = np.zeros(pop)       # fitness
    accuracy = np.zeros(pop)      # presición pred. correctas
    f1Score = np.zeros(pop)       # puntuación F1
    precision = np.zeros(pop)     # precisión pred. positivas
    recall = np.zeros(pop)        # exhaustividad
    mcc = np.zeros(pop)           # coeficiente de correlación Matthews
    errorRate = np.zeros(pop)     # rango de error
    totalSelected = np.zeros(pop) # cantidad de carac. seleccionadas

    # - primera población (todas las características)
    poblacion = np.ones((pop, dim))
    print("Primera población generada (aleatoria).")
    maxDiversidad = diversidadHussain(poblacion)

    # - fitness de la primera población
    for i in range(pop):
        fitness[i], accuracy[i], f1Score[i], precision[i], recall[i], mcc[i], errorRate[i], totalSelected[i] = fs.fitness(poblacion[i])

    # - definir mejor solución y mejor fitness
    orden = fitness.argsort(axis=0)
    mejorIdx = orden[0] # index del mejor fitness

    fitnessRanking = np.zeros(pop)  # ranking con los mejores fitness
    fitnessRanking = fitness[orden] 
    bestFitness = fitnessRanking[0] # mejor fitness

    solutionsRanking = np.zeros(pop)          # ranking con las mejores soluciones
    solutionsRanking = poblacion[orden, :]    
    bestSolution = solutionsRanking[0].copy() # mejor solución

    print("- Mejor solución: ")
    print(bestSolution)
    print("- Mejor fit: " + str(bestFitness))

    # medidas de la mejor solución
    bestAccuracy = accuracy[mejorIdx]
    bestF1Score = f1Score[mejorIdx]
    bestPrecision = precision[mejorIdx]
    bestRecall = recall[mejorIdx]
    bestMcc = mcc[mejorIdx]
    bestErrorRate = errorRate[mejorIdx]
    bestTFS = totalSelected[mejorIdx]

    print("- Desempeño del clasificador")
    print("Accuracy: " + str(bestAccuracy)
          + "\nF1 Score: " + str(bestF1Score)
          + "\nPrecision: " + str(bestPrecision)
          + "\nRecall" + str(bestRecall)
          + "\nMCC: " + str(bestMcc)
          + "\nError Rate: " + str(bestErrorRate)
          + "\nCant. Features Selected: " + str(bestTFS))

    # INICIO DE LA OPTIMIZACIÓN ----------------------------------------------------------------------------------
    print(" \n- OPTIMIZACIÓN -")
    time1 = time.time()
    binSol = poblacion.copy() # copia de la problación (funciona como "solución anterior")

    # instanciar metaheurística
    if (mh == 'MFO'):
        mfo = MothFlame(pop, dim, maxIter, ub, lb)
        flames = solutionsRanking.copy()
        flamesFit = fitnessRanking.copy()
    elif (mh == 'AVOA'):
        avoa = AfricanVultures(pop, dim, ub, lb, maxIter, paramMH)
    else:
        print("\nNo se ha encontrado una metaheurística con el nombre " + mh + ".")
        return -1, [], []
    
    # inicializar curva de convergencia
    convergenceCurve = np.zeros(shape=(maxIter))

    # Inicializar grafico de exploración vs explotación
    graficoExpl = np.zeros(shape=(maxIter,2))

    # Inicializar graficos de performance
    graficoPerf = np.zeros(shape=(7,maxIter))

    # Iteración principal
    for iteration in range(maxIter):
        print("\n- Iteración " + str(iteration+1) + " -")

        # Segunda y tercera iteración (perturvar población)
        if( mh == 'MFO' ):
            poblacion, flames, flamesFit = mfo.iterar(iteration, poblacion, fitness, flames, flamesFit)
        if( mh == 'AVOA' ):
            poblacion = avoa.iterar(poblacion, fitness, solutionsRanking)
        
        poblacion = np.clip(poblacion, lb, ub)
        
        # Procesamiento FS
        for i in range(pop):
            
            # binarizar
            poblacion[i] = binarizacion.aplicarBinarizacion(poblacion[i], funcTransf, funcBin, bestSolution, binSol[i].tolist())

            # comprobar factibilidad
            while not ins.esFactible(poblacion[i]): # mientras sea infactible
                print("* Solución Infactible *")
                poblacion[i] = fs.nuevaSolucion(1)  # genero una nueva
            
            # calcular fitness
            fitness[i], accuracy[i], f1Score[i], precision[i], recall[i], mcc[i], errorRate[i], totalSelected[i] = fs.fitness(poblacion[i])

            # actualizamos el ranking
            if (fitness[i] < bestFitness):
                # - agregamos el nuevo mejor
                fitnessRanking = np.insert(fitnessRanking, 0, fitness[i]) 
                solutionsRanking = np.insert(solutionsRanking, 0, poblacion[i], axis=0)

                # - quitamos el último
                fitnessRanking = fitnessRanking[:-1]
                solutionsRanking = solutionsRanking[:-1,:]

        
        # copiar la poblacion binarizada
        binSol = poblacion.copy()

        # verificar mejor fitness
        bestIndex = np.argsort(fitness)[0] # index del mejor fitness de la población actual
        if (fitness[bestIndex] < bestFitness):
            # tomar valores del nuevo mejor fitness
            bestFitness = fitness[bestIndex]
            bestSolution = poblacion[bestIndex].copy()
            bestAccuracy = accuracy[bestIndex]
            bestF1Score = f1Score[bestIndex]
            bestPrecision = precision[bestIndex]
            bestRecall = recall[bestIndex]
            bestMcc = mcc[bestIndex]
            bestErrorRate = errorRate[bestIndex]
            bestTFS = totalSelected[bestIndex]

        print("Mejor fitness de la iteración: " + str(fitness[bestIndex]))
        print("Mejor fitness histórico: " + str(bestFitness))

        # calcular diversidad y porcentajes XPL XPT de la iteración
        divIter = diversidadHussain(poblacion)

        if maxDiversidad < divIter:
            maxDiversidad = divIter

        XPL , XPT, state = porcentajesXPLXPT(divIter, maxDiversidad)

        # guardar para el gráfico
        graficoExpl[iteration][0] = XPL
        graficoExpl[iteration][1] = XPT

        # curva de convergencia
        convergenceCurve[iteration] = bestFitness

        # graficos de performance
        graficoPerf[0][iteration] = accuracy[bestIndex]
        graficoPerf[1][iteration] = f1Score[bestIndex]
        graficoPerf[2][iteration] = precision[bestIndex]
        graficoPerf[3][iteration] = recall[bestIndex]
        graficoPerf[4][iteration] = mcc[bestIndex]
        graficoPerf[5][iteration] = errorRate[bestIndex]
        graficoPerf[6][iteration] = totalSelected[bestIndex]

    time2 = time.time()
    tiempoEjec = time2 - time1
    
    performance = [bestAccuracy,bestF1Score,bestPrecision,bestRecall,bestMcc,bestErrorRate,bestTFS,tiempoEjec]

    return bestFitness, bestSolution, performance, convergenceCurve, graficoExpl, graficoPerf