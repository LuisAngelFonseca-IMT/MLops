from EDA import EDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sys
from loadParams import load_params

def dividir_datos(datos, test_size=0.3, validation_size=0.5):
        X = datos.drop(columns=["Status", 'ID'])
        y = datos["Status"]

        # División en entrenamiento (70%), validación (15%) y prueba (15%)
        X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_test1, y_test1, test_size=validation_size, random_state=42, stratify=y_test1)

        #Explorar los datos de entrenamiento
        EDA().boxPlotNumericas(X_train)
        EDA().histogramasNumericas(X_train)
        EDA().countPlotCategoricas(X_train)

        labelEncoder = LabelEncoder()

        y_preprocesado_df = pd.DataFrame(
            labelEncoder.fit_transform(y_train), 
            columns=["Status"],
            index=y_train.index
        )

        EDA().mapaDeCorrelacion(X_train,y_preprocesado_df,"Status")
        EDA().boxPlotVSTarget(X_train,y_train,"Status")

        return X_test,y_test,X_val,y_val,X_train,y_train

if __name__ == '__main__':
        parametros = load_params()
        base_path=parametros["data"]["basePath"]
        nombre_archivo=parametros["data"]["cirrhosisNombreArchivo"]
        csv_path=base_path+nombre_archivo+'.csv'
        datos=pd.read_csv(csv_path)
        X_test, y_test, X_val, y_val, X_train, y_train = dividir_datos(datos)
        X_test.to_csv(base_path+"X_test.csv",index=False)
        y_test.to_csv(base_path+"y_test.csv",index=False)
        X_val.to_csv(base_path+"X_val.csv",index=False)
        y_val.to_csv(base_path+"y_val.csv",index=False)
        X_train.to_csv(base_path+"X_train.csv",index=False)
        y_train.to_csv(base_path+"y_train.csv",index=False)

