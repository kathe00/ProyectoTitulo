"""
FEATURE SELECTION PROBLEM - Módulo de funciones específicas del problema

 Variables:
    - totalData: datos de la instancia
    - totalClass: clases de la instancia
    - totalFeatures: cantidad de características de la instancia
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from KNN.KNN import KNN


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
    
    def nuevaSolucion(self, tamPop): # binaria
        return np.random.randint(low=0, high=2, size = (tamPop, self.totalFeatures))
    
    def escalarData(self):
        escalador = preprocessing.MinMaxScaler()
        self.trainingData = escalador.fit_transform(self.trainingData)
        self.testingData = escalador.fit_transform(self.testingData)
    
    def dividirData(self, data):
        self.trainingData, self.testingData, self.trainingClass, self.testingClass  = train_test_split(
            data,
            self.totalClass,
            test_size=0.25,
            random_state=0
        )
    
    def fitness(self, sol):
        # instanciar KNN
        knn = KNN(self.k)

        # tomar solo las variables seleccionadas
        seleccion = np.where(sol == 1)[0]
        datos = self.totalData.iloc[:, seleccion]
        #print("\nfsproblem")
        #print(sol)
        #print(seleccion)
        #print(self.trainingData)

        # dividir datos (training-testing) y escalar valores
        self.dividirData(datos)
        self.escalarData()

        # probar KNN
        accuracy, f1Score, presicion, recall, mcc = knn.test(self.trainingData, self.testingData, self.trainingClass, self.testingClass)

        # calcular fitness
        errorRate = np.round((1 - accuracy), decimals=3)
        fitness = np.round(( self.gamma * errorRate ) + ( ( 1 - self.gamma ) * ( len(seleccion) / self.totalFeatures ) ), decimals=3)

        return fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, len(seleccion)
    
