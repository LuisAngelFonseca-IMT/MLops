import mlflow

def buscarCorrida(nombre_experimento):
    runs = mlflow.search_runs(experiment_names=[])
    print(runs)