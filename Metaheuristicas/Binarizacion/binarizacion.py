"""
BINARIZACIÓN - funciones de transferencia y binarización

 Input:
    - ind: individuo (vector de nros reales a ser binarizado)
    - trasnferFunction: nombre de la función de transferencia a utilizar
    - binarizationFunction: nombre del tipo de binarización a utilizar
    - bestSolutionBin: vector binario que representa la mejor solución encontrada
    - indBin: individuo binario, solución anteror
 Output:
    - individuoBin: individuo actual binarizado
"""

import random
import numpy as np
from scipy import special as scyesp

def aplicarBinarizacion(ind, transferFunction, binarizationFunction, bestSolutionBin, indBin):
    individuoBin = np.zeros(len(ind))

    for i in range(len(ind)): # recorrer solución binarizando
        step1 = transferir(transferFunction, ind[i])
        individuoBin[i] = binarizar(binarizationFunction, step1, bestSolutionBin[i], indBin[i])

    return np.array(individuoBin)

def transferir(transferFunction, dimension):

    if transferFunction == "S1":
        return np.divide( 1 , ( 1 + np.exp( -2 * dimension ) ) )
    
    if transferFunction == "S2":
        return np.divide( 1 , ( 1 + np.exp( -1 * dimension ) ) )
    
    if transferFunction == "S3":
        return np.divide( 1 , ( 1 + np.exp( np.divide( ( -1 * dimension ) , 2 ) ) ) )
    
    if transferFunction == "S4":
        return np.divide( 1 , ( 1 + np.exp( np.divide( ( -1 * dimension ) , 3 ) ) ) )
    
    if transferFunction == "V1":
        return np.abs( scyesp.erf( np.divide( np.sqrt( np.pi ) , 2 ) * dimension ) )
    
    if transferFunction == "V2":
        return np.abs( np.tanh( dimension ) )
    
    if transferFunction == "V3":
        return np.abs( np.divide( dimension , np.sqrt( 1 + np.power( dimension , 2 ) ) ) )
    
    if transferFunction == "V4":
        return np.abs( np.divide( 2 , np.pi ) * np.arctan( np.divide( np.pi , 2 ) * dimension ) )
    
    if transferFunction == "X1":
        return np.divide( 1 , ( 1 + np.exp( 2 * dimension ) ) )
    
    if transferFunction == "X2":
        return np.divide( 1 , ( 1 + np.exp( dimension ) ) )
    
    if transferFunction == "X3":
        return np.divide( 1 , ( 1 + np.exp( np.divide( dimension , 2 ) ) ) )
    
    if transferFunction == "X4":
        return np.divide( 1 , ( 1 + np.exp( np.divide( dimension , 3 ) ) ) )
    
    if transferFunction == "Z1":
        return np.power( ( 1 - np.power( 2 , dimension ) ) , 0.5 )
    
    if transferFunction == "Z2":
        return np.power( ( 1 - np.power( 5 , dimension ) ) , 0.5 )
    
    if transferFunction == "Z3":
        return np.power( ( 1 - np.power( 8 , dimension ) ) , 0.5 )
    
    if transferFunction == "Z4":
        return np.power( ( 1 - np.power( 20 , dimension ) ) , 0.5 )


def binarizar(binarizationFunction, step1, bestBin, indBin):

    if binarizationFunction == 'STD':
        rand = random.uniform(0.0, 1.0)
        binario = 0
        if rand <= step1:
            binario = 1
        return binario
    
    if binarizationFunction == 'COM':
        rand = random.uniform(0.0, 1.0)
        binario = 0
        if rand <= step1:
            if indBin == 1:
                binario = 0
            if indBin == 0:
                binario =  1
        return binario
    
    if binarizationFunction == 'PS':
        alpha = 1/3
        binario = 0
        if alpha < step1 and step1 <= ( ( 1/2 ) * ( 1 + alpha ) ):
            binario = indBin
        if step1 > ( ( 1/2 ) * ( 1 + alpha ) ):
            binario = 1
        return binario
    
    if binarizationFunction == 'ELIT':
        rand = random.uniform(0.0, 1.0)
        binario = 0
        if rand < step1:
            binario = bestBin
        return binario
    
