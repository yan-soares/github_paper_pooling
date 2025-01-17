import pandas as pd
import ast
import os

# Função para processar strings representando dicionários com `eval`
def parse_dict_with_eval(value):
    try:
        if isinstance(value, str):
            # Substituir 'np.float64' por 'float' para compatibilidade
            value = value.replace('np.float64', 'float')
            return eval(value)
        return {}
    except Exception as e:
        return {}

# Carregar o arquivo CSV
main_path = 'experimentos_21122024_4sessoes/results_original/classification_original/'
results_path_acc = 'experimentos_21122024_4sessoes/resultados_processados/classification_acc'
results_path_devacc = 'experimentos_21122024_4sessoes/resultados_processados/classification_devacc'

arquivos_csv = [f for f in os.listdir(main_path) if f.endswith('.csv')]
ordem_colunas = ['Modelo', 'Pooling', 'epochs', 'qtd_layers', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']
for arquivo in arquivos_csv:
    caminho_arquivo = os.path.join(main_path, arquivo)

    data = pd.read_csv(caminho_arquivo, encoding="utf-8", on_bad_lines="skip")

    # Identificar as colunas de tarefas
    task_columns = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']

    # Criar os dicionários para armazenar os dados de `devacc` e `acc`
    devacc_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling']}
    acc_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling']}

    # Processar os dados para extrair `devacc` e `acc`
    for task in task_columns:
        devacc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('devacc', None))
        acc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('acc', None))

    # Converter os dicionários em DataFrames
    devacc_table = pd.DataFrame(devacc_data)
    devacc_table['epochs'] = arquivo.split('_')[-2][6:]
    devacc_table['qtd_layers'] = arquivo.split('_')[-1].split('.')[0][9:]

    acc_table = pd.DataFrame(acc_data)
    acc_table['epochs'] = arquivo.split('_')[-2][6:]
    acc_table['qtd_layers'] = arquivo.split('_')[-1].split('.')[0][9:]

    devacc_table = devacc_table[ordem_colunas]
    acc_table = acc_table[ordem_colunas]

    devacc_table.to_csv(os.path.join(results_path_devacc, arquivo.split('.csv')[0] + '_processado_devacc.csv'))
    acc_table.to_csv(os.path.join(results_path_acc, arquivo.split('.csv')[0]) + '_processado_acc.csv')
    
   

