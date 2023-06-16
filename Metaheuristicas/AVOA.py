"""
AFRICAN VULTURES OPTIMIZATION (AVOA) - implementación de la metaheurística

 Variables:
    - nsa: nro de agentes de búsqueda (buitres)
    - max_iter: máximo de iteraciones
    - ub : valores máximos de cada variable
    - lb : valores mínimos de cada variable
    - dim: dimensión (nro variables)
    - rp1: probabilidad 1 (para definir qué etapa de exploración seguir)
    - rp2: probabilidad 2 (para definir qué etapa seguir durante la fase 1 de explotación)
    - rp3: probabilidad 3 (para definir qué etapa seguir durante la fase 2 de explotación)
    - l  : parámetro l (para definir a cual buitre seguir)
    - w  : parámetro w (probabilidad de pasar entre explr y explt. A mayor w, mayor exploración)
"""
import numpy as np
import random
import math

class AfricanVultures():
    def __init__(self, nsa, dim, ub, lb, max_iter, param):
        # parámetros
        self.nsa = nsa                        # nro de agentes de búsqueda (buitres)
        self.dim = dim                        # dimensión (nro variables)
        self.ub = ub                          # valores máximos de cada variable
        self.lb = lb                          # valores mínimos de cada variable
        self.max_iter = max_iter              # máximo de iteraciones
        self.rp1 = float(param.split(',')[0]) # probabilidad 1
        self.rp2 = float(param.split(',')[1]) # probabilidad 2
        self.rp3 = float(param.split(',')[2]) # probabilidad 3
        self.l = float(param.split(',')[3])   # parámetro l (para definir a cual buitre seguir)
        self.w = int(param.split(',')[4])     # parámetro w (probabilidad de pasar entre explr y explt) (a mayor w, más explora)

    def levyFlight(self, d):  
        """
        LevyFlight definition by
        https://github.com/angelinbeni/African-Vulture-Optimization

        """
        beta = 3/2
        sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = np.random.randn(1,d)*sigma
        v = np.random.randn(1,d)
        step = u/abs(v)**(1/beta)
        o = step
        return o

    def aux_div(self, dividendo, divisor): # para divisiones por 0
        if divisor == 0:
            division = dividendo / 0.000000001
        else:
            division = dividendo / divisor
        return division
    
    def iterar(self, poblacion, fitness, rankSol):
        # definir al primer y segundo mejor buitre
        best1 = rankSol[0]
        best2 = rankSol[1]

        # recorrer población 
        for i in range(self.nsa):
            # calcular p y elegir uno de los mejores buitres
            p = fitness[i] / sum(fitness)

            if p >= self.l:
                bbuitre = best1.copy()
            else:
                bbuitre = best2.copy()

            # actualizar t y z, y calcular F
            h = random.uniform(-2,2)
            #print("h: " + str(h))
            t = h * ( math.sin( (math.pi / 2) * (i+1 / self.max_iter) )**self.w + math.cos( (math.pi / 2) * (i+1 / self.max_iter) ) - 1 )
            #print("t: " + str(t))
            z = random.uniform(-1,1)
            #print("z: " + str(z))
            rand = random.random()
            #print("rand: " + str(rand))
            f = ( ( (2 * rand) + 1 ) * z * ( 1 - (i / self.max_iter) ) ) + t
            #print("F: " + str(f))

            if abs(f) >= 1:
                # calcular p1
                p1 = random.random()
                if p1 >= self.rp1:
                    # Actualizar posición según eq. 3.10 y 3.11
                    x = 2 * random.random()
                    #print("x: " + str(x))
                    for j in range(self.dim):
                        di = abs((x * bbuitre[j]) - poblacion[i][j])
                        poblacion[i][j] = bbuitre[j] - (di * f)
                    #print("Exploración E1")
                    #print(poblacion[i])
                else:
                    # Actualizar posición según eq. 3.12
                    rand2 = random.random()
                    for j in range(self.dim):
                        rand3 = random.random()
                        poblacion[i][j] = bbuitre[j] - f + (rand2 * ((self.ub[j] - self.lb[j]) * rand3 + self.lb[j]))
                    #print("Exploración E2")
                    #print(poblacion[i])
            else:
                if abs(f) >= 0.5:
                    # buitre saciado
                    p2 = random.random()
                    if p2 >= self.rp2:
                        # Actualizar posición según eq. 3.13
                        rand4 = random.random()
                        x = 2 * random.random()
                        for j in range(self.dim):
                            di = abs((x * bbuitre[j]) - poblacion[i][j])
                            dt = bbuitre[j] - poblacion[i][j]
                            poblacion[i][j] = (di * (f + rand4)) - dt
                        #print("Explotación F1E1")
                        #print(poblacion[i])
                    else:
                        # Actualizar posición según eq. 3.17
                        rand5 = random.random()
                        rand6 = random.random()
                        for j in range(self.dim):
                            s1 = bbuitre[j] * ((rand5 * poblacion[i][j]) / 2 * math.pi) * math.cos(poblacion[i][j])
                            s2 = bbuitre[j] * ((rand6 * poblacion[i][j]) / 2 * math.pi) * math.sin(poblacion[i][j])
                            poblacion[i][j] = bbuitre[j] - (s1 + s2)
                        #print("Explotación F1E2")
                        #print(poblacion[i])
                else:
                    # buitre hambriento
                    p3 = random.random()
                    if p3 >= self.rp3:
                        # Actualizar posición según eq. 3.13
                        for j in range(self.dim):
                            aux_dividendo = best1[j] * poblacion[i][j]
                            aux_divisor = best1[j] - pow(poblacion[i][j],2)
                            aux_division = self.aux_div(aux_dividendo, aux_divisor)
                                
                            a1 = best1[j] - ( aux_division * f )

                            aux_dividendo = best2[j] * poblacion[i][j]
                            aux_divisor = best2[j] - pow(poblacion[i][j],2)
                            aux_division = self.aux_div(aux_dividendo, aux_divisor)

                            a2 = best2[j] - ( aux_division * f )

                            poblacion[i][j] = (a1 + a2) / 2
                        #print("Explotación F2E1")
                        #print(poblacion[i])
                    else:
                        # Actualizar posición según eq. 3.17
                        lf = self.levyFlight(self.dim)
                        for j in range(self.dim):
                            dt = bbuitre[j] - poblacion[i][j]
                            poblacion[i] = bbuitre[j] - ( abs(dt) * f * lf )
                        #print("Explotación F2E2")
                        #print(poblacion[i])
            
        return poblacion
