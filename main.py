"""
PROYECTO DE TÍTULO - INGENIERÍA DE EJECUCIÓN INFORMÁTICA 2023
Katherine Sepúlveda

Módulo principal para la ejecución de experimentos.
Parámetros:
    - repeticiones: número de repeticiones del experimento
    - problema: nombre que identifica al problema
                'FS'  -> Feature Selection
                'SCP' -> Set Covering Problem
                'BM'  -> Benchmark
    - instancia: nombre de la instancia del problema
    - paramProblema: parámetros específicos del problema
    - datosMFO: datos específicos de Moth Flame Optimization -> 'nombre,población,lowerBound,upperBound'
                - MFO no tiene parámetros específicos
    - datosAVOA: datos específicos de African Vultures Optimization -> 'nombre,población,lowerBound,upperBound'
    - paramAVOA: parámetros específicos de AVOA -> 'rp1,rp2,rp3,l,w'
"""
from Solver.solver import FeatureSelection
from Solver.solver import BenchMark
from Solver.solver import SetCoveringProblem

# ----------------------------------------------------------------------------------- #
# ---------------------------------- EXPERIMETOS ------------------------------------ #

# -- PARÁMETROS CONFIGURABLES

# - Parámetros Experimento
repeticiones = 1
problema = 'SCP'

# - Parámetros Problema 
# - descomentar solo el correspondiente al problema a resolver

## Feature Selection
#instancia = 'ionosphere'
#paramProblem = '10,3,0.99,S2,ELIT' # -> 'maxIter,k,gamma,funcTransferencia,tipoBinarización'

## Benchmark
#instancia = 'F3'
#paramProblem = '600,30' # -> 'maxIter,cantDimensiones'

# SetCovering
instancia = 'scp41'
paramProblem = '100,S1,ELIT' # -> 'maxIter,funcTransferencia,tipoBinarización'

# - Parámetros Metaheurísticas
datosMFO = 'MFO,50,-5,5'

datosAVOA = 'AVOA,20,-5,5'
paramAVOA = '0.5,0.8,0.5,0.7,4'

# EJECUCIÓN
for i in range(repeticiones):
    if (problema == 'FS'):
        FeatureSelection(instancia, datosMFO, '', paramProblem) # MFO
        #FeatureSelection(instancia, datosAVOA, paramAVOA, paramProblem) # AVOA
    if (problema == 'BM'):
        #BenchMark(instancia, datosMFO, '', paramProblem) # MFO
        BenchMark(instancia, datosAVOA, paramAVOA, paramProblem) # AVOA
    if (problema == 'SCP'):
        SetCoveringProblem(instancia, datosMFO, '', paramProblem) # MFO
        #SetCoveringProblem(instancia, datosAVOA, paramAVOA, paramProblem) # AVOA

