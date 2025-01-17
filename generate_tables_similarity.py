import pandas as pd
import os
import shutil

###CREATE PATHS
path_arquivos_originais = "experiments_results"
similarity_file_name = [f for f in os.listdir(path_arquivos_originais) if f.startswith('si_') and f.endswith('.csv')][0]
experiment_name = '_'.join(similarity_file_name.split('.csv')[0].split('_'))
experiment_path = path_arquivos_originais + "/" + experiment_name
path_similarity_pearson = path_arquivos_originais + "/" + experiment_name + "/" + "similarity_pearson"
path_similarity_spearman = path_arquivos_originais + "/" + experiment_name + "/" + "similarity_spearman"
os.makedirs(path_similarity_pearson, exist_ok=True)
os.makedirs(path_similarity_spearman, exist_ok=True)

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
ordem_colunas_si = ['Modelo', 'Pooling', 'out_emb_size', 'epochs', 'qtd_layers', 'nhid',  'STS12', 'STS13', 'STS14', 'STS15', 'STS16']

###similarity
caminho_arquivo_si = os.path.join(path_arquivos_originais, similarity_file_name)
data = pd.read_csv(caminho_arquivo_si, encoding="utf-8", on_bad_lines="skip")

task_columns = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']

spearman_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'out_emb_size': data['out_emb_size'], 'nhid': data['Nhid']}
pearson_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'out_emb_size': data['out_emb_size'], 'nhid': data['Nhid']}

for task in task_columns:
    if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        pearson_data[task] = data[task].apply(lambda x: (parse_dict_with_eval(x).get('pearson', None).get('mean', None)) * 100)
        spearman_data[task] = data[task].apply(lambda x: (parse_dict_with_eval(x).get('spearman', None).get('mean', None)) * 100)

spearman_table = pd.DataFrame(spearman_data)
spearman_table['epochs'] = similarity_file_name.split('_')[-1].split('.')[0].split('-')[-1]
spearman_table['qtd_layers'] = 12

pearson_table = pd.DataFrame(pearson_data)
pearson_table['epochs'] = similarity_file_name.split('_')[-1].split('.')[0].split('-')[-1]
pearson_table['qtd_layers'] = 12

spearman_table = spearman_table[ordem_colunas_si]
pearson_table = pearson_table[ordem_colunas_si]

spearman_table['Avg'] = spearman_table[task_columns].mean(axis=1)
pearson_table['Avg'] = pearson_table[task_columns].mean(axis=1)

spearman_table.to_csv(os.path.join(path_similarity_spearman, similarity_file_name.split('.csv')[0] + '_processado_spearman.csv'))
pearson_table.to_csv(os.path.join(path_similarity_pearson, similarity_file_name.split('.csv')[0]) + '_processado_pearson.csv')

shutil.move(caminho_arquivo_si, experiment_path + "/" + similarity_file_name)