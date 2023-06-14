"""
FEATURE SELECTION PROBLEM - Módulo de funciones específicas del problema

 Variables:
    - totalData: datos de la instancia
    - totalClass: clases de la instancia
    - totalFeatures: cantidad de características de la instancia (dimensión)
    - trainingData: conjunto de datos para entrenar el clasificador
    - trainingClass: conjunto de clases (relativas a los datos) para entrenar el clasificador
    - testingData: conjunto de datos para probar el clasificador
    - testingClass: conjunto de clases para probar el clasificador
    - gamma: balance de importancia entre cant. de características seleccionadas y presición del clasificador
    - k: cantidad de vecinos para KNN 
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from FeatureSelection.Clasificadores.KNN import KNN


class FeatureSelection:
    def __init__(self, datos, clases, totalFeatures, gamma, k):
        self.totalData = datos
        self.totalClass = clases
        self.totalFeatures = totalFeatures
        self.trainingData = None
        self.trainingClass = None
        self.testingData = None
        self.testingClass = None
        self.gamma = gamma
        self.k = k

    
    def getTotalFeatures(self):
        return self.totalFeatures

    def setTotalFeatures(self, total):
        self.totalFeatures = total
    
    def nuevaSolucion(self, tamPop): # genera una solución binaria (tamaño de la población x dimensión)
        return np.random.randint(low=0, high=2, size = (tamPop, self.totalFeatures))
    
    def escalarData(self): # genera el escalador y normaliza los valores del dataset
        escalador = preprocessing.MinMaxScaler()
        self.trainingData = escalador.fit_transform(self.trainingData)
        self.testingData = escalador.fit_transform(self.testingData)
    
    def dividirData(self, data): # divide el set de datos entre training y testing
        self.trainingData, self.testingData, self.trainingClass, self.testingClass  = train_test_split(
            data,
            self.totalClass,
            test_size=0.25,
            random_state=0
        )
    
    def fitness(self, sol): # cálculo del fitness para Feature Selection
        # instanciar KNN
        knn = KNN(self.k)

        # tomar solo las variables seleccionadas
        selection = np.where(sol == 1)[0]
        datos = self.totalData.iloc[:, selection]

        # dividir datos (training-testing) y escalar valores
        self.dividirData(datos)
        self.escalarData()

        # probar KNN
        accuracy, f1Score, presicion, recall, mcc = knn.test(self.trainingData, self.testingData, self.trainingClass, self.testingClass)

        # calcular fitness
        errorRate = np.round((1 - accuracy), decimals=3)
        fitness = np.round(( self.gamma * errorRate ) + ( ( 1 - self.gamma ) * ( len(selection) / self.totalFeatures ) ), decimals=3)

        return fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, len(selection)
    
