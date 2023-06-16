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
problema = 'FS' # -> FS | SCP | BM 

# --- Parámetros Problema ---
# - descomentar solo el correspondiente al problema a resolver

# Feature Selection
instancia = 'hill-valley' # -> ionosphere | hill-valley | Qsar | ARCENE | DOROTHEA | MADELON
paramProblem = '10,10,0.9999,V4,STD' # -> 'maxIter,k,gamma,funcTransferencia,tipoBinarización'

## Benchmark
#instancia = 'F2' # -> F1 a F8
#paramProblem = '1500,15' # -> 'maxIter,cantDimensiones'

## SetCovering
#instancia = 'scp41' # -> scp41 | scp51 | scp61
#paramProblem = '20,V4,COM' # -> 'maxIter,funcTransferencia,tipoBinarización'

# --- Parámetros Metaheurísticas ---
datosMFO = 'MFO,15,-10,10' # -> 'nombre,población,lowerBound,upperBound'

datosAVOA = 'AVOA,15,-2,2' # -> 'nombre,población,lowerBound,upperBound'
paramAVOA = '0.9,0.5,0.7,2,3' # -> 'rp1,rp2,rp3,l,w'

# -- EJECUCIÓN
for i in range(repeticiones):
    # - Directorio
    fecha = crearDirectorio(problema)

    if (problema == 'FS'):
        FeatureSelection(instancia, datosMFO, '', paramProblem, fecha)          # MFO
        FeatureSelection(instancia, datosAVOA, paramAVOA, paramProblem, fecha)  # AVOA
    if (problema == 'BM'):
        BenchMark(instancia, datosMFO, '', paramProblem, fecha)                 # MFO
        BenchMark(instancia, datosAVOA, paramAVOA, paramProblem, fecha)         # AVOA
    if (problema == 'SCP'):
        SetCoveringProblem(instancia, datosMFO, '', paramProblem, fecha)         # MFO
        SetCoveringProblem(instancia, datosAVOA, paramAVOA, paramProblem, fecha) # AVOA

# -------------------

