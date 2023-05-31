import time
import numpy as np
from BenchMark.benchProblem import fitness as f
from Metaheuristicas.MFO import MothFlame


def solverBench(pop, dim, maxIter, ub, lb, function):
    # INICIALIZACIÓN
    print("-----------------------------------------------------------------")
    print("Resolviendo función ["+ function + "]")
    print("-----------------------------------------------------------------")
    print(" \n- INICIALIZACIÓN -")
    print("Tamaño de población: " + str(pop))

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

        fitness[i] = f(function, poblacion[i])
    
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

    # INICIO DE LA OPTIMIZACIÓN ----------------------------------------------------------------------------------
    print(" \n- OPTIMIZACIÓN -")
    time1 = time.time()

    # - Moth Flame Optimization
    # instanciar metaheurística
    mfo = MothFlame(pop, dim, maxIter, ub, lb)

    # Iteración principal
    for iteration in range(maxIter):

        # Segunda y tercera iteración (perturvar población)
        poblacion, solutionsRanking, fitnessRanking = mfo.iterar(iteration, poblacion, fitness, solutionsRanking, fitnessRanking)

        # cáculo del fitness
        for i in range(poblacion.__len__()):
            for j in range(dim):
                poblacion[i, j] = np.clip(poblacion[i, j], lb[j], ub[j])            

            fitness[i] = f(function, poblacion[i])
        

        # obtener rankings
        solutionsRanking, fitnessRanking = mfo.ordenarFlamas(solutionsRanking, poblacion, fitnessRanking, fitness)
        bestFitness = fitnessRanking[0]
        bestSolution = solutionsRanking[0].copy()
        print("Iteracion " + str(iteration))
        print(" - Best Fitness: " + str(bestFitness))

    print(" \n- FIN OPTIMIZACIÓN -")
    print(" Best Fitness: " + str(bestFitness))
    print(" Best Solution: " + str(bestSolution))

    time2 = time.time()
    tiempoEjec = time2 - time1
    print("Tiempo de ejecución: " + str(tiempoEjec))