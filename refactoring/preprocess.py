import numpy as np
import pandas as pd
import sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from loadParams import load_params


def crearPipelinePreprocesamientoX():
        columnas_numericas_skew = ["Bilirubin", "Cholesterol", "Copper", "Alk_Phos", "SGOT", "Tryglicerides"]
        columnas_numericas_no_skew = ["N_Days", "Age", "Albumin", "Platelets", "Prothrombin"]
        columnas_categoricas = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"]
        pipeline_numerico_skew = Pipeline(steps=[
            ('imputador', SimpleImputer(strategy='median')),
            ('log_transform', FunctionTransformer(np.log1p, validate=False)),
            ('escalador', StandardScaler())
        ])
        pipeline_numerico_no_skew = Pipeline(steps=[
            ('imputador', SimpleImputer(strategy='median')),
            ('escalador', StandardScaler())
        ])
        pipeline_categorico = Pipeline(steps=[
            ('imputador', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformador_columnas = ColumnTransformer(transformers=[
            ('num_skew', pipeline_numerico_skew, columnas_numericas_skew),
            ('num_no_skew', pipeline_numerico_no_skew, columnas_numericas_no_skew),
            ('cat', pipeline_categorico, columnas_categoricas)
        ], remainder='passthrough')

        return transformador_columnas

def crearPipelinePreprocesamientoy():
    label_encoder = LabelEncoder()
    return label_encoder


if __name__ == '__main__':
    parametros = load_params()
    base_path=parametros["data"]["basePath"]
    preprocesado_path = parametros["data"]["preprocesdePath"]


    X_test = pd.read_csv(base_path+"X_test.csv")
    y_test = pd.read_csv(base_path+"y_test.csv").values.ravel()
    X_val = pd.read_csv(base_path+"X_val.csv")
    y_val = pd.read_csv(base_path+"y_val.csv").values.ravel()
    X_train = pd.read_csv(base_path+"X_train.csv")
    y_train = pd.read_csv(base_path+"y_train.csv").values.ravel()

    x_preprocess_pipeline = crearPipelinePreprocesamientoX()
    y_preprocess_pipeline = crearPipelinePreprocesamientoy()
    x_preprocess_pipeline.fit(X_train)
    y_preprocess_pipeline = crearPipelinePreprocesamientoy()
    y_preprocess_pipeline.fit(y_train)

    X_test = x_preprocess_pipeline.transform(X_test)
    y_test = y_preprocess_pipeline.transform(y_test)
    X_val = x_preprocess_pipeline.transform(X_val)
    y_val = y_preprocess_pipeline.transform(y_val)
    X_train = x_preprocess_pipeline.transform(X_train)
    y_train = y_preprocess_pipeline.transform(y_train)

    pd.DataFrame(X_train).to_csv(preprocesado_path+"X_train.csv", index=False)
    pd.DataFrame(y_train).to_csv(preprocesado_path+"y_train.csv", index=False)
    pd.DataFrame(X_val).to_csv(preprocesado_path+"X_val.csv", index=False)
    pd.DataFrame(y_val).to_csv(preprocesado_path+"y_val.csv", index=False)
    pd.DataFrame(X_test).to_csv(preprocesado_path+"X_test.csv", index=False)
    pd.DataFrame(y_test).to_csv(preprocesado_path+"y_test.csv", index=False)


