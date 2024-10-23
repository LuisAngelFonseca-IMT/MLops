from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle
import sys
import pandas as pd
from loadParams import load_params

def entrenar_modelo(model,param_grid,X_train,y_train):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train.ravel())

        return grid_search.best_estimator_

def modeloRlog(X_train,y_train,param_grid_logistico):
        modelo_logistico = LogisticRegression(max_iter=5000)
        
        modelo = modelo_logistico
        modelo = entrenar_modelo(modelo,param_grid_logistico,X_train,y_train)
        return modelo


def modeloXGboost(X_train,y_train,param_grid_xgboost):
        modelo_xgboost = XGBClassifier(use_label_encoder=True, eval_metric='logloss')
        modelo = modelo_xgboost
        modelo = entrenar_modelo(modelo,param_grid_xgboost,X_train,y_train)
        return modelo

def salvarModelo(modelo,path):
        print(path)
        pickle.dump(modelo, open(path, 'wb'))

if __name__ == '__main__':
        parametros = load_params()
        preprocesado_path = parametros["data"]["preprocesdePath"]
        modelos_path = parametros["modelos"]["path"]
        param_grid_xgboost = parametros["modelos"]["grid_xgboost"]
        param_grid_logistico = parametros["modelos"]["grid_logistico"]

        X_train = pd.read_csv(preprocesado_path+"X_train.csv")
        y_train = pd.read_csv(preprocesado_path+"y_train.csv").values.ravel()

        modeloLogRegresion = modeloRlog(X_train,y_train,param_grid_logistico)
        modeloXG = modeloXGboost(X_train,y_train,param_grid_xgboost)

        salvarModelo(modeloLogRegresion,modelos_path+"modeloRegLog.sav")
        salvarModelo(modeloXG,modelos_path+"modeloXGBoost.sav")
        
        
        