import time
import numpy as np
from SetCovering.SCProblem import SetCovering
from Metaheuristicas.MFO import MothFlame
from Metaheuristicas.AVOA import AfricanVultures
from SetCovering.leerInstancia import leerInstancia
from Metaheuristicas.Binarizacion import binarizacion
from Solver.diversidad import porcentajesXPLXPT, diversidadHussain


def solverSCP(instancia, datosMH, paramMH, paramProblem):
    # PARÁMETROS
    mh = datosMH.split(',')[0]              # nombre Metaheurística
    pop = int(datosMH.split(',')[1])        # tamaño de la población
    maxIter = int(paramProblem.split(',')[0]) # cantidad de iteraciones
    lb = int(datosMH.split(',')[2])         # lower bound
    ub = int(datosMH.split(',')[3])         # upper bound
    funcTransf = paramProblem.split(',')[1]   # función de transferencia
    funcBin = paramProblem.split(',')[2]      # función de binarización

    inst = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/" + instancia + ".txt"

    # INICIALIZACIÓN
    print("-----------------------------------------------------------------")
    print("Resolviendo función [ "+ instancia + " ]")
    print("-----------------------------------------------------------------")
    print(" \n- INICIALIZACIÓN -")
    print("Metaheurística: " + mh
          + "\nPoblación: " + str(pop)
          + "\nIteraciones: " + str(maxIter))

    # - leer la instancia
    instance = leerInstancia(inst)
    dim = instance.columnas   # dimensión del problema

    # - primera población binaria
    poblacion = np.random.randint(low=0, high=2, size = (pop, dim))
    print("Primera población generada.")

    # - cálculo de la diversidad
    maxDiversidad = diversidadHussain(poblacion)
    XPL , XPT, state = porcentajesXPLXPT(maxDiversidad, maxDiversidad)

    # - vector para fitness
    fitness = np.zeros(pop)

    # - instanciar límites
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    # - inicializar SCP
    scp = SetCovering(pop, ub, lb, instance.matriz_A, instance.costos, instance.columnas)

    # - y factibilidad
    for i in range(pop):
        while not scp.solFactible(poblacion[i])[0]: # mientras sea infactible
            #print("* Solución Infactible *")
            poblacion[i] = scp.reparar(poblacion[i])  # reparo

    # - cálculo del fitness
    fitness = scp.funcionObjetivo(poblacion)
    
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

    # Inicializar curva de convergencia
    convergenceCurve = np.zeros(shape=(maxIter))

    # Inicializar grafico de exploración vs explotación
    graficoExpl = np.zeros(shape=(maxIter,2))

    # Iteración principal
    for iteration in range(maxIter):
        print("\n- Iteración " + str(iteration+1) + " -")

        # Segunda y tercera iteración (perturvar población)
        if( mh == 'MFO' ):
            poblacion, flames, flamesFit = mfo.iterar(iteration, poblacion, fitness, flames, flamesFit)
        if( mh == 'AVOA' ):
            poblacion = avoa.iterar(poblacion, fitness, solutionsRanking)
        
        poblacion = np.clip(poblacion, lb, ub)

        for i in range(pop):
            # binarizar
            poblacion[i] = binarizacion.aplicarBinarizacion(poblacion[i], funcTransf, funcBin, bestSolution, binSol[i].tolist())

            # comprobar factibilidad
            while not scp.solFactible(poblacion[i])[0]: # mientras sea infactible
                #print("* Solución Infactible *")
                poblacion[i] = scp.reparar(poblacion[i])  # reparo
            
        # calcular fitness
        fitness = scp.funcionObjetivo(poblacion)
        
        # obtener rankings de la iteracion
        order = fitness.argsort(axis=0)
        fitnessRanking = fitness[order]
        solutionsRanking = poblacion[order, :]

        # verificar mejor fitness
        if (fitnessRanking[0] < bestFitness):
            # tomar valores del nuevo mejor fitness
            bestFitness = fitnessRanking[0]
            bestSolution = solutionsRanking[0].copy()
        
        print(fitness)
        print("Mejor fitness de la iteración: " + str(fitnessRanking[0]))
        print("Mejor fitness histórico: " + str(bestFitness))

        # calcular diversidad y porcentajes XPL XPT de la iteración
        divIter = diversidadHussain(poblacion)

        if maxDiversidad < divIter:
            maxDiversidad = divIter

        XPL , XPT, state = porcentajesXPLXPT(divIter, maxDiversidad)

        # guardar para el gráfico
        graficoExpl[iteration][0] = XPL
        graficoExpl[iteration][1] = XPT

        # guardo el valor para la curva de convergencia
        convergenceCurve[iteration] = bestFitness

    time2 = time.time()
    tiempoEjec = time2 - time1
    
    performance = [tiempoEjec]

    return bestFitness, bestSolution, performance, convergenceCurve, graficoExpl