"""
MOTH FLAME OPTIMIZATION (MFO) - implementación de la metaheurística

 Variables:
    - nsa: nro de agentes de búsqueda (polillas)
    - maxIter: máximo de iteraciones
    - ub : valores máximos de cada variable
    - lb : valores mínimos de cada variable
    - dim: dimensión (nro variables)
    - b  : constante definida en 1
"""
import numpy as np

class MothFlame():
    def __init__(self, nsa, dim, maxIter, ub, lb):
        # parámetros
        self.nsa = nsa
        self.maxIter = maxIter
        self.ub = ub
        self.lb = lb
        self.dim = dim
        self.b = 0.75
    
    def ordenarFlamas(self, flames, mothPos, flamesFit, mothFit):
        # se comparan los valores de las polillas y las flamas
        # las flamas son las mejores soluciones históricas

        # juntar
        doublePop = np.vstack((flames, mothPos))
        doubleFit = np.hstack((flamesFit, mothFit))

        # ordenar
        order = doubleFit.argsort(axis=0)
        doubleFit = doubleFit[order]
        doublePop = doublePop[order, :]

        # definir las llamas
        flam = doublePop[:self.nsa, :]
        flamFit = doubleFit[:self.nsa]

        return flam, flamFit
        
    def iterar(self, iteration, poblacion, fitness, solRank, fitRank):

        mothPos = poblacion
        mothFit = fitness

        # Número de llamas, Eq. en fig. 3.12 del paper
        flameNo = int(np.ceil(self.nsa - (iteration + 1) * ((self.nsa - 1) / self.maxIter)))
        
        # Asegurar que los valores de las polillas esten dentro de los márgenes
        mothPos = np.clip(mothPos, self.lb, self.ub)

        # Si es la primera generación de polillas las llamas son una copia ordenada de las polillas
        # En cada iteracion, las llamas llevan el ranking de las mejores sol. encontradas
        flames = np.copy(solRank)
        flamesFit = np.copy(fitRank)

        if iteration != 0:
            flames, flamesFit = self.ordenarFlamas(flames, mothPos, flamesFit, mothFit)

        ## realizar la perturvación
        # r decrece linealmente de -1 a -2 para calcular t en Eq. de fig. (3.10)
        r = -1 + (iteration + 1) * ((-1) / self.maxIter)
        # distancia D entre llamas y polillas
        distanceToFlames = np.abs(flames - mothPos)
        # actualizar valor de t
        t = (r - 1) * np.random.rand(self.nsa, self.dim) + 1
        
        # Actualizar la posición de las polillas en relación a sus llamas correspondientes.
        # Si el nro de la polilla es mayor al nro total de llamas, entonces se
        # actualiza en relación a la última llama. 

        # cortar según la cantidad de llamas de la iteración
        temp1 = flames[:flameNo, :]
        # las ultimas posiciones se llenan con la ultima llama
        temp2 = flames[flameNo - 1, :] * np.ones(shape=(self.nsa - flameNo, self.dim))
        temp2 = np.vstack((temp1, temp2))

        # mover las polillas
        mothPos = distanceToFlames * np.exp(self.b * t) * np.cos(t * 2 * np.pi) + temp2

        return mothPos, flames, flamesFit

