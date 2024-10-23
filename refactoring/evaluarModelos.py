import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import sys
import pandas as pd
from loadParams import load_params
import mlflow
import json

metricasFinales={"entrenamiento":{},"validacion":{},"prueba":{}}

class evaluarModelo():
    def __init__(self,modelo,X_train,y_train,X_val,y_val,X_test,y_test,tipoModelo):
         self.modelo = modelo
         self.X_train = X_train
         self.y_train=y_train
         self.X_val=X_val
         self.y_val=y_val
         self.X_test=X_test
         self.y_test=y_test
         self.tipoModelo = tipoModelo
         self.metricas = {"entrenamiento":{},"validacion":{},"prueba":{}}


    def evaluar(self):
        print(f'Evaluado Modelo {self.tipoModelo}')
        y_pred_test = self.modelo.predict(self.X_test)
        y_pred_val = self.modelo.predict(self.X_val)
        y_pred_train = self.modelo.predict(self.X_train)
        self.matriz_de_confuision(self.y_train,y_pred_train,"entrenamiento",self.tipoModelo)
        self.matriz_de_confuision(self.y_val,y_pred_val,"validacion",self.tipoModelo)
        self.matriz_de_confuision(self.y_test,y_pred_test,"prueba",self.tipoModelo)
        self.precision_train,self.recall_train,self.f1_train = self.metricasDeRendimiento(self.y_train,y_pred_train,"entrenamiento")
        self.precision_val,self.recall_val,self.f1_val =self.metricasDeRendimiento(self.y_val,y_pred_val,"validacion")
        self.precision_test,self.recall_test,self.f1_test =self.metricasDeRendimiento(self.y_test,y_pred_test,"prueba")
        metricasFinales["entrenamiento"][f"{self.tipoModelo}_precision_entrenamiento"] = self.precision_train
        metricasFinales["entrenamiento"][f"{self.tipoModelo}_recall_entrenamiento"] = self.recall_train
        metricasFinales["entrenamiento"][f"{self.tipoModelo}_f1_entrenamiento"] = self.f1_train
        metricasFinales["validacion"][f"{self.tipoModelo}_precision_validacion"] = self.precision_val
        metricasFinales["validacion"][f"{self.tipoModelo}_recall_validacion"] = self.recall_val
        metricasFinales["validacion"][f"{self.tipoModelo}_f1_validacion"] = self.f1_val
        metricasFinales["prueba"][f"{self.tipoModelo}_precision_prueba"] = self.precision_test
        metricasFinales["prueba"][f"{self.tipoModelo}_recall_prueba"] = self.recall_test
        metricasFinales["prueba"][f"{self.tipoModelo}_f1_prueba"] = self.f1_test
        self.logModeloAMLFlow()

    @staticmethod
    def matriz_de_confuision(y,ypred,tipoDeSetDeDatos,tipoModelo):
        # Matriz de confusión
        cm = confusion_matrix(y, ypred)
        figure = plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matriz de Confusión - Conjunto de {tipoDeSetDeDatos}")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        figure.savefig(f"./docs/confusion_matrixes/{tipoModelo}_{tipoDeSetDeDatos}.png")
        mlflow.log_figure(figure,f"{tipoModelo}_{tipoDeSetDeDatos}.png")

    @staticmethod
    def metricasDeRendimiento(y,ypred,tipoDeSetDeDatos):
        # Cálculo de métricas con zero_division
        precision = precision_score(y, ypred, average='weighted', zero_division=0)
        recall = recall_score(y, ypred, average='weighted', zero_division=0)
        f1 = f1_score(y, ypred, average='weighted', zero_division=0)

        # Mostrar resultados
        print(f"Precisión de conjunto de {tipoDeSetDeDatos}: {precision:.2f}")
        print(f"Recall de conjunto de {tipoDeSetDeDatos}: {recall:.2f}")
        print(f"Puntuación de conjunto de {tipoDeSetDeDatos}: {f1:.2f}")

        return precision,recall,f1

    def logModeloAMLFlow(self):
        hyperParametrosDelModelo = self.modelo.get_params()
        for parametro in hyperParametrosDelModelo:
                mlflow.log_param(f"{parametro} {self.tipoModelo}",hyperParametrosDelModelo[parametro])
        mlflow.log_metric(f'precision entrenamiento {self.tipoModelo}',self.precision_train)
        mlflow.log_metric(f'recall entrenamiento {self.tipoModelo}',self.recall_train)
        mlflow.log_metric(f'f1 entrenamiento {self.tipoModelo}',self.f1_train)
        mlflow.log_metric(f'precision validacion {self.tipoModelo}',self.precision_val)
        mlflow.log_metric(f'recall validacion {self.tipoModelo}',self.recall_val)
        mlflow.log_metric(f'f1 validacion {self.tipoModelo}',self.f1_val)
        mlflow.log_metric(f'precision prueba {self.tipoModelo}',self.precision_test)
        mlflow.log_metric(f'recall prueba {self.tipoModelo}',self.recall_test)
        mlflow.log_metric(f'f1 prueba {self.tipoModelo}',self.f1_test)
        mlflow.sklearn.log_model(self.modelo, f"{self.tipoModelo}_model")



if __name__ == '__main__':
    parametros = load_params()
    preprocesado_path = parametros["data"]["preprocesdePath"]
    modelos_path = parametros["modelos"]["path"]

    X_train = pd.read_csv(preprocesado_path+"X_train.csv")
    y_train = pd.read_csv(preprocesado_path+"y_train.csv")
    X_val = pd.read_csv(preprocesado_path+"X_val.csv")
    y_val = pd.read_csv(preprocesado_path+"y_val.csv")
    X_test = pd.read_csv(preprocesado_path+"X_test.csv")
    y_test = pd.read_csv(preprocesado_path+"y_test.csv")

    with open(modelos_path+'modeloRegLog.sav', 'rb') as f:
        modeloRLog = pickle.load(f)

    with open(modelos_path+'modeloXGBoost.sav', 'rb') as f:
        modeloXGBoost = pickle.load(f)

    mlflow.set_tracking_uri(parametros['mlflow']['tracking_uri'])
    experiment = mlflow.set_experiment(parametros['mlflow']['experiment_name'])

    

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        evaluarLogReg = evaluarModelo(modeloRLog,X_train,y_train,X_val,y_val,X_test,y_test,"LogRegresion")
        evaluarXGBoost = evaluarModelo(modeloXGBoost,X_train,y_train,X_val,y_val,X_test,y_test,"XGBoost")
        evaluarLogReg.evaluar()
        evaluarXGBoost.evaluar()

    mlflow.end_run()

    json_object = json.dumps(metricasFinales, indent=4)
    with open("metricas.json", "w") as outfile:
        outfile.write(json_object)


    
    