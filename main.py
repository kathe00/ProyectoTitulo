"""
PROYECTO DE TÍTULO - INGENIERÍA DE EJECUCIÓN INFORMÁTICA 2023
Katherine Sepúlveda
"""

from Solver.solverFS import solverFS
from Solver.solverFS import solverBench
from Metaheuristicas.AVOA import AfricanVultures

# - PARÁMETROS - #
instancia = 'ionosphere'
tamPop = 10
maxIter = 30
low = 0
high = 2
k = 3
gamma = 0.99
esquema = ['S2', 'ELIT']

# MFO
ubMFO = 100
lbMFO = 10

# AVOA
# mh = AfricanVultures(3, 4, high, low, maxIter, 0.5, 0.5, 0.5, 0.5, 3)
# pop = [[0,1,1,0],[0,0,1,1],[1,1,1,1]]
# rank = [[0,0,1,1],[0,1,1,0],[1,1,1,1]]
# mh.iterar(pop, [0.5,0.7,0.1],rank)

#solverFS(instancia, tamPop, k, gamma, maxIter, esquema, high, low)


