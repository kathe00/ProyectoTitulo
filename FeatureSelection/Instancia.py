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
        return (np.sum(sol) > 0)

    def leerInstancia(self, instancia):
        if instancia == 'ionosphere':
            
            # leemos la instancia
            instancia = instancia + ".data"
            dataset = pd.read_csv('./FeatureSelection/Instancias/' + instancia, header=None)

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

            
"""
        if instancia == 'ionosphere':
            instancia = instancia + ".data"
            classPosition = 34
            dataset = pd.read_csv('Problem/FS/Instances/' + instancia, header=None)
            clases = dataset.iloc[:,classPosition] 
            clases = clases.replace({
                'b':0,
                'g':1
            })
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')

        elif instancia == 'sonar':
            instancia = instancia+".all-data"
            classPosition = 60
            dataset = pd.read_csv('Problem/FS/Instances/'+instancia, header=None)
            clases = dataset.iloc[:,classPosition]
            clases = clases.replace({
                'R':0,
                'M':1
            })
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')
            
        elif instancia == 'Hill-Valley-with-noise.data' or instancia == 'Hill-Valley-without-noise.data':
            classPosition = 100
            dataset = pd.read_csv('Problem/FS/Instancias/'+instancia, header=None)
            clases = dataset.iloc[:,classPosition]
            clases = clases.values

            datos = dataset.drop(dataset.columns[classPosition],axis='columns')
"""
