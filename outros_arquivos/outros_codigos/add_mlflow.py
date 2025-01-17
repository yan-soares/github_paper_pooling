import pandas as pd
import mlflow
import os

#Definir o nome do experimento
experiment_name_acc = "Resultados Classificação 21122024 - ACC"
experiment_name_devacc = "Resultados Classificação 21122024 - DEVACC"

results_path_acc = 'experimentos_21122024_4sessoes/resultados_processados/classification_acc/'
results_path_devacc = 'experimentos_21122024_4sessoes/resultados_processados/classification_devacc/'

arquivos_csv_acc = [f for f in os.listdir(results_path_acc) if f.endswith('.csv')]
arquivos_csv_devacc = [f for f in os.listdir(results_path_devacc) if f.endswith('.csv')]

#add resultados acc
mlflow.set_experiment(experiment_name_acc)
for arquivo in arquivos_csv_acc:
    caminho_arquivo = os.path.join(results_path_acc, arquivo)
    df = pd.read_csv(caminho_arquivo, encoding="utf-8", on_bad_lines="skip")

   # Registrar os dados do DataFrame no MLflow
    with mlflow.start_run(run_name=arquivo):
        # Iterar pelas linhas do DataFrame
        # Registrar cada coluna como um conjunto de parâmetros/métricas
        for coluna in df.columns:
            valores = df[coluna].tolist()  # Obter os valores da coluna como uma lista
            
            if pd.api.types.is_numeric_dtype(df[coluna]):
                # Registrar a média dos valores como métrica (exemplo simplificado)
                mlflow.log_metric(f"mean_{coluna}", sum(valores) / len(valores))
            else:
                # Registrar valores não numéricos como parâmetros (convertidos em string)
                mlflow.log_param(f"valores_{coluna}", ", ".join(map(str, valores)))
        
        # Salvar o DataFrame completo como artefato
        mlflow.log_artifact(caminho_arquivo, artifact_path="dados_csv")

    print("DataFrame registrado no MLflow com sucesso!")
    
   

