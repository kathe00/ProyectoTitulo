import numpy as np
import random


class SetCovering():

  def __init__(self, nsa, ub, lb, A, costos,cols):
    # parámetros
    self.nsa = nsa  # nro agentes de busqueda
    self.ub = ub  # valores máximos de cada variable
    self.lb = lb  # valores mínimos de cada variable
    self.matriz_A = A # matriz de cobertura
    self.costos = costos # vector de costos
    self.columnas = cols # columnas = cant variables

# --- Solución Inicial ---

  def solucionInicial(self):
    # Genera una solución binaria inicial con valores random para cada variable
    pop = np.random.randint(low=0, high=2, size = (self.nsa, self.dim))
    return pop

# --- Funcion Objetivo ---

  def funcionObjetivo(self, sol):
    # se calcula el fitness con el producto punto entre cada fila (solución)
    # y el vector de los costos de asignación
    fit = np.zeros(self.nsa)
    for i in range(self.nsa):
      fit[i] = np.dot(sol[i,:],self.costos)

    return fit

# --- Comprobación de Factibilidad ---

  def solFactible(self, sol):
    # con el producto punto entre la matriz de cobertura y la solución
    # se obtiene un vector que indica la correcta cobertura de cada restricción
    aux = np.dot(self.matriz_A, sol)
    aux2 = np.argwhere(aux == 0) # arreglo con las restricciones no cubiertas
    # si este vector no está vacío, es infactible
    if (np.size(aux2) != 0): return False, aux

    return True, aux  # si el arreglo está vacío, es factible

# --- reparación ---

  def reparar(self, solution):
    sol =  np.reshape(solution, (self.columnas,))
    set = self.matriz_A
    aux = np.dot(self.matriz_A, sol)

    nz = np.argwhere(aux == 0)                              # Obtengo las restricciones no cubiertas
    id_nz = np.random.choice(nz[0])                         # Selecciono una restricción no cubierta aleatoriamente
    idxRestriccion = np.argwhere((set[id_nz,:]) > 0)        # Obtengo la lista de subsets que cubren la zona seleccionada
    cost = np.take(self.costos, idxRestriccion)
    a = np.argmin(cost)                                     # Obtengo el/los subset que tiene/n el costo mas bajo 
    idxMenorPeso = idxRestriccion[a]
    sol[np.random.choice(idxMenorPeso)] = 1                 # Asigno 1 a ese subset

    return sol
