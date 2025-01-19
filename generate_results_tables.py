import argparse
import pandas as pd
import os
import shutil

MAIN_PATH = "tables_main_experiments"
cl_paths = [p for p in os.listdir(MAIN_PATH) if p.startswith('cl_')]
si_paths = [p for p in os.listdir(MAIN_PATH) if p.startswith('si_')]

columns_tasks_cl = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
columns_tasks_si = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']

main_colunas = ['Modelo', 'Pooling', 'Epochs', 'out_vec_size', 'qtd_layers', 'Nhid']
ordem_colunas_cl = main_colunas + columns_tasks_cl 
ordem_colunas_si = main_colunas + columns_tasks_si 

def parse_dict_with_eval(value):
    try:
        if isinstance(value, str):
            value = value.replace('np.float64', 'float')
            return eval(value)
        return {}
    except Exception as e:
        return {}

def tables_classification(cl_paths, columns_tasks_cl, ordem_colunas_cl):
    for clp in cl_paths:
        path_cl = MAIN_PATH + '/' + clp
        path_cl_acc = path_cl + "/" + "cl_acc"
        path_cl_devacc = path_cl + "/" + "cl_devacc"
        os.makedirs(path_cl_acc, exist_ok=True)
        os.makedirs(path_cl_devacc, exist_ok=True)

        cl_file_name = [f for f in os.listdir(path_cl) if f.endswith('.csv')][0]

        caminho_arquivo_cl = os.path.join(path_cl, cl_file_name)
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

def tables_similarity():
    path_arquivos_originais = "experiments_results"
    similarity_file_name = [f for f in os.listdir(path_arquivos_originais) if f.startswith('si_') and f.endswith('.csv')][0]
    experiment_name = '_'.join(similarity_file_name.split('.csv')[0].split('_'))
    experiment_path = path_arquivos_originais + "/" + experiment_name
    path_similarity_pearson = path_arquivos_originais + "/" + experiment_name + "/" + "similarity_pearson"
    path_similarity_spearman = path_arquivos_originais + "/" + experiment_name + "/" + "similarity_spearman"
    os.makedirs(path_similarity_pearson, exist_ok=True)
    os.makedirs(path_similarity_spearman, exist_ok=True)
    
    caminho_arquivo_si = os.path.join(path_arquivos_originais, similarity_file_name)
    data = pd.read_csv(caminho_arquivo_si, encoding="utf-8", on_bad_lines="skip")

    spearman_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'out_emb_size': data['out_emb_size'], 'nhid': data['Nhid']}
    pearson_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'out_emb_size': data['out_emb_size'], 'nhid': data['Nhid']}

    for task in columns_tasks_si:
        pearson_data[task] = data[task].apply(lambda x: (parse_dict_with_eval(x).get('pearson', None).get('mean', None)) * 100)
        spearman_data[task] = data[task].apply(lambda x: (parse_dict_with_eval(x).get('spearman', None).get('mean', None)) * 100)

    spearman_table = pd.DataFrame(spearman_data)[ordem_colunas_si]
    pearson_table = pd.DataFrame(pearson_data)[ordem_colunas_si] 

    spearman_table['Avg'] = spearman_table[columns_tasks_si].mean(axis=1)
    pearson_table['Avg'] = pearson_table[columns_tasks_si].mean(axis=1)

    spearman_table.to_csv(os.path.join(path_similarity_spearman, similarity_file_name.split('.csv')[0] + '_processado_spearman.csv'))
    pearson_table.to_csv(os.path.join(path_similarity_pearson, similarity_file_name.split('.csv')[0]) + '_processado_pearson.csv')

    shutil.move(caminho_arquivo_si, experiment_path + "/" + similarity_file_name)

def main():
    parser = argparse.ArgumentParser(description="SentEval Experiments")
    parser.add_argument("--task_type", type=str, required=True, default="classification", help="Tipo de tarefa (classification ou similarity)")
    args = parser.parse_args()

    task_type_args = args.task_type 

    if task_type_args == "classification":
        tables_classification(cl_paths, columns_tasks_cl, ordem_colunas_cl)

    elif task_type_args == "similarity":
        tables_similarity(si_paths, columns_tasks_si, ordem_colunas_si)

if __name__ == "__main__":
    print('ok')
    #main()