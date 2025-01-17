import pandas as pd
import os
import shutil

###CREATE PATHS
path_arquivos_originais = "experimentos_teste"
classification_file_name = [f for f in os.listdir(path_arquivos_originais) if f.startswith('classification')][0]
similarity_file_name = [f for f in os.listdir(path_arquivos_originais) if f.startswith('similarity')][0]
log_file_name = [f for f in os.listdir(path_arquivos_originais) if f.startswith('log')][0]
experiment_name = '_'.join(classification_file_name.split('.csv')[0].split('_')[1:])
experiment_path = path_arquivos_originais + "/" + experiment_name
path_classification_acc = path_arquivos_originais + "/" + experiment_name + "/" + "classification_acc"
path_classification_devacc = path_arquivos_originais + "/" + experiment_name + "/" + "classification_devacc"
path_similarity_pearson = path_arquivos_originais + "/" + experiment_name + "/" + "similarity_pearson"
path_similarity_spearman = path_arquivos_originais + "/" + experiment_name + "/" + "similarity_spearman"
os.makedirs(path_classification_acc, exist_ok=True)
os.makedirs(path_classification_devacc, exist_ok=True)
os.makedirs(path_similarity_pearson, exist_ok=True)
os.makedirs(path_similarity_spearman, exist_ok=True)
shutil.move(os.path.join(path_arquivos_originais, log_file_name), experiment_path + "/" + log_file_name)

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
ordem_colunas_cl = ['Modelo', 'Pooling', 'epochs', 'qtd_layers', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']
ordem_colunas_si = ['Modelo', 'Pooling', 'epochs', 'qtd_layers', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']

###CLASSIFICATION
caminho_arquivo_cl = os.path.join(path_arquivos_originais, classification_file_name)
data = pd.read_csv(caminho_arquivo_cl, encoding="utf-8", on_bad_lines="skip")

task_columns = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']
task_columns_without_SST5 = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']

devacc_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling']}
acc_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling']}

for task in task_columns:
    devacc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('devacc', None))
    acc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('acc', None))

devacc_table = pd.DataFrame(devacc_data)
devacc_table['epochs'] = classification_file_name.split('_')[-2][6:]
devacc_table['qtd_layers'] = classification_file_name.split('_')[-1].split('.')[0][9:]

acc_table = pd.DataFrame(acc_data)
acc_table['epochs'] = classification_file_name.split('_')[-2][6:]
acc_table['qtd_layers'] = classification_file_name.split('_')[-1].split('.')[0][9:]

devacc_table = devacc_table[ordem_colunas_cl]
acc_table = acc_table[ordem_colunas_cl]

devacc_table['Avg_with_SST5'] = devacc_table[task_columns].mean(axis=1)
devacc_table['Avg_without_SST5'] = devacc_table[task_columns_without_SST5].mean(axis=1)
acc_table['Avg_with_SST5'] = acc_table[task_columns].mean(axis=1)
acc_table['Avg_without_SST5'] = acc_table[task_columns_without_SST5].mean(axis=1)

devacc_table.to_csv(os.path.join(path_classification_devacc, classification_file_name.split('.csv')[0] + '_processado_devacc.csv'))
acc_table.to_csv(os.path.join(path_classification_acc, classification_file_name.split('.csv')[0]) + '_processado_acc.csv')

shutil.move(caminho_arquivo_cl, experiment_path + "/" + classification_file_name)

###SIMILARITY
caminho_arquivo_si = os.path.join(path_arquivos_originais, similarity_file_name) 
data = pd.read_csv(caminho_arquivo_si, encoding="utf-8", on_bad_lines="skip")

task_columns =  ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']

pearson_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling']}
spearman_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling']}

for task in task_columns:
    if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        pearson_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('pearson', None).get('mean', None))
        spearman_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('spearman', None).get('mean', None))
    else:
        pearson_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('pearson', None))
        spearman_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('spearman', None))

pearson_table = pd.DataFrame(pearson_data)
pearson_table['epochs'] = similarity_file_name.split('_')[-2][6:]
pearson_table['qtd_layers'] = similarity_file_name.split('_')[-1].split('.')[0][9:]

spearman_table = pd.DataFrame(spearman_data)
spearman_table['epochs'] = similarity_file_name.split('_')[-2][6:]
spearman_table['qtd_layers'] = similarity_file_name.split('_')[-1].split('.')[0][9:]

pearson_table = pearson_table[ordem_colunas_si]
spearman_table = spearman_table[ordem_colunas_si]

pearson_table['Avg'] = pearson_table[task_columns].mean(axis=1)
spearman_table['Avg'] = spearman_table[task_columns].mean(axis=1)

pearson_table.to_csv(os.path.join(path_similarity_pearson, similarity_file_name.split('.csv')[0] + '_processado_pearson.csv'))
spearman_table.to_csv(os.path.join(path_similarity_spearman, similarity_file_name.split('.csv')[0]) + '_processado_spearman.csv')

shutil.move(caminho_arquivo_si, experiment_path + "/" + similarity_file_name)