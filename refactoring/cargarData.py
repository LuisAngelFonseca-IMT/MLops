from EDA import EDA
import pandas as pd
import sys
from loadParams import load_params

def cargar_datos(csvpath):
        datos = pd.read_csv(csvpath)
        EDA.explorarDataFrame(datos)
        return datos

if __name__ == '__main__':
        parametros = load_params()
        base_path=parametros["data"]["basePath"]
        path_descarga=parametros["data"]["downloadPath"]
        nombre_archivo=parametros["data"]["cirrhosisNombreArchivo"]
        archivo_csv_descargado=path_descarga+nombre_archivo+".csv"
        archivo_csv_base = base_path+nombre_archivo+".csv"
        datos = cargar_datos(archivo_csv_descargado)
        datos.to_csv(archivo_csv_base,index=False)
