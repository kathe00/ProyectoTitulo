"""
LECTURA DE INSTANCIA

 Variables:
    - datos: matriz con los datos de la instancia
    - clases: vector con las clases de la instancia, relativas a los datos
"""

import numpy as np
import pandas as pd

class Instancia:

    datos = []
    clases = []
    
    def __init__(self):
        self.datos
        self.clases

    def getDatos(self):
        return self.datos
    
    def getClases(self):
        return self.clases
    
    def esFactible(self, sol):
        return (np.sum(sol) > 0) # que haya al menos un elemento seleccionado

    def leerInstancia(self, instancia):
        if instancia == 'ionosphere':
            
            # leemos la instancia
            instancia = "ionosphere.data"
            dataset = pd.read_csv('./FeatureSelection/Instancias/Ionosphere/' + instancia, header=None)

            # separamos las clases de los datos (última columna de cada fila)
            cantAtributos = 34
            clases = dataset.iloc[:,cantAtributos]

            # pasamos los valores de las clases a binario
            clases = clases.replace({
                'b':0,
                'g':1
            })
            clases = clases.values 

            # y quitamos la última columna para quedar solo con los datos
            datos = dataset.drop(dataset.columns[cantAtributos],axis='columns')

            # guardamos la instancia
            self.clases = clases
            self.datos = datos

        elif instancia == 'hill-valley':

            instancia = "Hill-Valley-without-noise.data"
            dataset = pd.read_csv('./FeatureSelection/Instancias/HillValley/' + instancia, header=None)

            cantAtributos = 101
            clases = dataset.iloc[:,cantAtributos-1]
            clases = clases.values # clases binarias
            datos = dataset.drop(dataset.columns[cantAtributos-1],axis='columns')

            # guardamos la instancia
            self.clases = clases
            self.datos = datos
            
        elif instancia == 'Qsar':

            instancia = "qsar_androgen_receptor.csv"
            dataset = pd.read_csv('./FeatureSelection/Instancias/Qsar/' + instancia, header=None, sep=';')
            print(dataset)

            cantAtributos = 1024
            clases = dataset.iloc[:,cantAtributos]
            clases = clases.replace({
                'negative':0,
                'positive':1
            })
            clases = clases.values 
            datos = dataset.drop(dataset.columns[cantAtributos],axis='columns')

            # guardamos la instancia
            self.clases = clases
            self.datos = datos

        elif instancia == 'ARCENE' or 'DOROTHEA' or 'MADELON':

            ruta = "./FeatureSelection/Instancias/" + instancia + "/" + instancia.lower()

            if instancia == 'ARCENE': cantAtributos = 10000
            elif instancia == 'DOROTHEA': cantAtributos = 100000
            else: cantAtributos = 500 # MADELON

            # juntamos los archvos de train y valid
            aux_dataset = pd.read_csv(ruta+'_train.data', header=None, sep=' ')
            aux_dataset2 = pd.read_csv(ruta+'_valid.data', header=None, sep=' ')
            dataset = pd.concat([aux_dataset,aux_dataset2])
            
            # quitamos la columna final (NaN)
            dataset = dataset.drop(dataset.columns[cantAtributos],axis='columns')

            # leemos las clases
            aux_clases = pd.read_csv(ruta+'_train.labels', header=None, sep=' ')
            aux_clases2 = pd.read_csv(ruta+'_valid.labels', header=None, sep=' ')
            clases = pd.concat([aux_clases,aux_clases2])

            clases = clases.iloc[:,0]
            clases = clases.replace({
                '-1':0,
                '1':1
            })
            clases = clases.values

            # guardamos la instancia
            self.clases = clases
            self.datos = dataset
            