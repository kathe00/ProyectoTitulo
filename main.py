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
from util import crearDirectorio

# ----------------------------------------------------------------------------------- #
# ---------------------------------- EXPERIMETOS ------------------------------------ #

# -- PARÁMETROS CONFIGURABLES

# --- Parámetros Experimento ---
repeticiones = 1
problema = 'FS'

# --- Parámetros Problema ---
# - descomentar solo el correspondiente al problema a resolver

# Feature Selection
instancia = 'ionosphere' # -> ionosphere | hill-valley | Qsar | ARCENE
paramProblem = '200,10,0.9999,V4,STD' # -> 'maxIter,k,gamma,funcTransferencia,tipoBinarización'

## Benchmark
#instancia = 'F3'
#paramProblem = '600,30' # -> 'maxIter,cantDimensiones'

## SetCovering
#instancia = 'scp51'
#paramProblem = '20,V4,COM' # -> 'maxIter,funcTransferencia,tipoBinarización'

# --- Parámetros Metaheurísticas ---
datosMFO = 'MFO,5,-1,1'

datosAVOA = 'AVOA,15,-1,1'
paramAVOA = '0.5,0.5,0.5,2,10' # -> 'rp1,rp2,rp3,l,w'

# -- Directorio
fecha = crearDirectorio()

# -- EJECUCIÓN
for i in range(repeticiones):
    if (problema == 'FS'):
        FeatureSelection(instancia, datosMFO, '', paramProblem, fecha) # MFO
        #FeatureSelection(instancia, datosAVOA, paramAVOA, paramProblem, fecha) # AVOA
    if (problema == 'BM'):
        #BenchMark(instancia, datosMFO, '', paramProblem, fecha) # MFO
        BenchMark(instancia, datosAVOA, paramAVOA, paramProblem, fecha) # AVOA
    if (problema == 'SCP'):
        #SetCoveringProblem(instancia, datosMFO, '', paramProblem, fecha) # MFO
        SetCoveringProblem(instancia, datosAVOA, paramAVOA, paramProblem, fecha) # AVOA

# -------------------

