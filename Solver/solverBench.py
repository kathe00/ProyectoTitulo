import time
import numpy as np
from BenchMark.benchProblem import fitness as f
from Metaheuristicas.MFO import MothFlame
from Metaheuristicas.AVOA import AfricanVultures


def solverBench(instancia, datosMH, paramMH, paramProblem):
    # PARÁMETROS
    funcion = instancia                     # función a resolver
    mh = datosMH.split(',')[0]              # nombre Metaheurística
    pop = int(datosMH.split(',')[1])        # tamaño de la población
    maxIter = int(paramProblem.split(',')[0]) # cantidad de iteraciones
    dim = int(paramProblem.split(',')[1])   # dimensión del problema
    lb = int(datosMH.split(',')[2])         # lower bound
    ub = int(datosMH.split(',')[3])         # upper bound

    # INICIALIZACIÓN
    print("-----------------------------------------------------------------")
    print("Resolviendo función [ "+ funcion + " ]")
    print("-----------------------------------------------------------------")
    print(" \n- INICIALIZACIÓN -")
    print("Metaheurística: " + mh
          + "\nPoblación: " + str(pop)
          + "\nIteraciones: " + str(maxIter))

    # - primera población 
    poblacion = np.random.uniform(low=lb, high=ub, size = (pop, dim))
    print("Primera población generada.")

    # - vector para fitness
    fitness = np.zeros(pop)

    # - instanciar límites
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # - cálculo del fitness
    for i in range(poblacion.__len__()):
        for j in range(dim):
            poblacion[i, j] = np.clip(poblacion[i, j], lb[j], ub[j])

        fitness[i] = f(funcion, poblacion[i])
    
    print("Fitness inicial calculado, valores:")
    print(fitness)

    # - definir mejor solución y fitness
    orden = fitness.argsort(axis=0)

    fitnessRanking = np.zeros(pop)
    fitnessRanking = fitness[orden] # fitness ordenados de mejor a peor
    bestFitness = fitnessRanking[0] # mejor fitness
    print("Mejor fit: " + str(bestFitness))

    solutionsRanking = np.zeros(pop)
    solutionsRanking = poblacion[orden, :]    # soluciones ordenadas de mejor a peor
    bestSolution = solutionsRanking[0].copy() # mejor solución
    print("Mejor solución: " + str(bestSolution))

    # INICIO DE LA OPTIMIZACIÓN ----------------------------------------------------------------------------------
    print(" \n- OPTIMIZACIÓN -")
    time1 = time.time()

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

    # Inicializar curva de convergencia
    convergenceCurve = np.zeros(shape=(maxIter))

    # Iteración principal
    for iteration in range(maxIter):
        print("\n- Iteración " + str(iteration+1) + " -")

        # Segunda y tercera iteración (perturvar población)
        if( mh == 'MFO' ):
            poblacion, flames, flamesFit = mfo.iterar(iteration, poblacion, fitness, flames, flamesFit)
        if( mh == 'AVOA' ):
            poblacion = avoa.iterar(poblacion, fitness, solutionsRanking)

        poblacion = np.clip(poblacion, lb, ub)
        
        # cáculo del fitness
        for i in range(poblacion.__len__()):
            for j in range(dim):
                poblacion[i, j] = np.clip(poblacion[i, j], lb[j], ub[j])            

            fitness[i] = f(funcion, poblacion[i])
        
        # obtener rankings de la iteracion
        order = fitness.argsort(axis=0)
        fitnessRanking = fitness[order]
        solutionsRanking = poblacion[order, :]

        # verificar mejor fitness
        if (fitnessRanking[0] < bestFitness):
            # tomar valores del nuevo mejor fitness
            bestFitness = fitnessRanking[0]
            bestSolution = solutionsRanking[0].copy()
        
        print("Mejor fitness de la iteración: " + str(fitnessRanking[0]))
        print("Mejor fitness histórico: " + str(bestFitness))
        
        convergenceCurve[iteration] = bestFitness

    time2 = time.time()
    tiempoEjec = time2 - time1
    
    performance = [tiempoEjec]

    return bestFitness, bestSolution, performance, convergenceCurve