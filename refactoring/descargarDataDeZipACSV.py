import os
from urllib.request import urlretrieve
from zipfile import ZipFile
import sys
from loadParams import load_params


class descargarDataDeZipACSV:
    def __init__(self,url,zippath,csvpath):
        self.url = url
        self.zippath = zippath
        self.csvpath = csvpath
        self.descargarDeUrl()
        self.extraerDeZip()
        self.borrarZip()

    def descargarDeUrl(self):
        urlretrieve(self.url,self.zippath)

    def extraerDeZip(self):
        with ZipFile(self.zippath) as zObject: 
                zObject.extractall(path=self.csvpath)
    
    def borrarZip(self):
        os.remove(self.zippath)

if __name__ == '__main__':
    parametros = load_params()
    url=parametros["data"]["url"]
    path_descarga=parametros["data"]["downloadPath"]
    nombre_archivo=parametros["data"]["cirrhosisNombreArchivo"]
    zippath=path_descarga+nombre_archivo+".zip"
    print(zippath)
    descargarDataDeZipACSV(url,zippath,path_descarga)