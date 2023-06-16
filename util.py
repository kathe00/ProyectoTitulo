"""
UTIL
Funciones generales
"""

from datetime import datetime
import os

def crearDirectorio(problema):
    # formato fecha
    fecha = datetime.now().strftime("%d%m%Y_%H%M%S")
    fecha = problema + "_" + fecha

    # crear nueva carpeta
    directorio = "./Resultados/" + fecha
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    return fecha