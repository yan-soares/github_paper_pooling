import pandas as pd
import os
import shutil

###CREATE PATHS
path_arquivos_originais = "experiments_results"
classification_file_name = [f for f in os.listdir(path_arquivos_originais) if f.startswith('cl_nhid-0') and f.endswith('.csv')][0]
experiment_name = '_'.join(classification_file_name.split('.csv')[0].split('_'))
experiment_path = path_arquivos_originais + "/" + experiment_name
path_classification_acc = path_arquivos_originais + "/" + experiment_name + "/" + "classification_acc"
path_classification_devacc = path_arquivos_originais + "/" + experiment_name + "/" + "classification_devacc"
os.makedirs(path_classification_acc, exist_ok=True)
os.makedirs(path_classification_devacc, exist_ok=True)

###FUNCOES
def parse_dict_with_eval(value):
    try:
        if isinstance(value, str):
            # Substituir 'np.float64' por 'float' para compatibilidade
            value = value.replace('np.float64', 'float')
            return eval(value)
        return {}
    except Exception as e:
        return {}
    
###VARIAVEIS
ordem_colunas_cl = ['Modelo', 'Pooling', 'out_emb_size', 'epochs', 'qtd_layers', 'nhid', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']

###CLASSIFICATION
caminho_arquivo_cl = os.path.join(path_arquivos_originais, classification_file_name)
data = pd.read_csv(caminho_arquivo_cl, encoding="utf-8", on_bad_lines="skip")

task_columns = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']
task_columns_without_SST5 = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']

devacc_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'out_emb_size': data['out_emb_size'], 'nhid': data['Nhid']}
acc_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'out_emb_size': data['out_emb_size'], 'nhid': data['Nhid']}

for task in task_columns:
    devacc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('devacc', None))
    acc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('acc', None))

devacc_table = pd.DataFrame(devacc_data)
devacc_table['epochs'] = classification_file_name.split('_')[-1].split('.')[0].split('-')[-1]
devacc_table['qtd_layers'] = 12

acc_table = pd.DataFrame(acc_data)
acc_table['epochs'] = classification_file_name.split('_')[-1].split('.')[0].split('-')[-1]
acc_table['qtd_layers'] = 12

devacc_table = devacc_table[ordem_colunas_cl]
acc_table = acc_table[ordem_colunas_cl]

devacc_table['Avg_with_SST5'] = devacc_table[task_columns].mean(axis=1)
devacc_table['Avg_without_SST5'] = devacc_table[task_columns_without_SST5].mean(axis=1)
acc_table['Avg_with_SST5'] = acc_table[task_columns].mean(axis=1)
acc_table['Avg_without_SST5'] = acc_table[task_columns_without_SST5].mean(axis=1)

devacc_table.to_csv(os.path.join(path_classification_devacc, classification_file_name.split('.csv')[0] + '_processado_devacc.csv'))
acc_table.to_csv(os.path.join(path_classification_acc, classification_file_name.split('.csv')[0]) + '_processado_acc.csv')

shutil.move(caminho_arquivo_cl, experiment_path + "/" + classification_file_name)