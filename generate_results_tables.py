import argparse
import pandas as pd
import os
import shutil

def parse_dict_with_eval(value):
    try:
        if isinstance(value, str):
            value = value.replace('np.float64', 'float')
            return eval(value)
        return {}
    except Exception as e:
        return {}

MAIN_PATH = "tables_main_experiments"
cl_files = [f for f in os.listdir(MAIN_PATH) if f.startswith('cl_') and f.endswith('.csv')]
si_files = [f for f in os.listdir(MAIN_PATH) if f.startswith('si_') and f.endswith('.csv')]

columns_tasks_cl = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
columns_tasks_si = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']

main_colunas = ['Modelo', 'Pooling', 'Epochs', 'out_vec_size', 'qtd_layers', 'Nhid']
ordem_colunas_cl = main_colunas + columns_tasks_cl 
ordem_colunas_si = main_colunas + columns_tasks_si 

def main():
    parser = argparse.ArgumentParser(description="SentEval Experiments")
    parser.add_argument("--task_type", type=str, required=True, default="classification", help="Tipo de tarefa (classification ou similarity)")
    args = parser.parse_args()

    task_type_args = args.task_type 

    if task_type_args == "classification":
        for file_cl in cl_files:
            experiment_name = '_'.join(file_cl.split('.csv')[0].split('_'))
            experiment_path = MAIN_PATH + "/" + experiment_name
            path_classification_acc = MAIN_PATH + "/" + experiment_name + "/" + "classification_acc"
            path_classification_devacc = MAIN_PATH + "/" + experiment_name + "/" + "classification_devacc"
            os.makedirs(path_classification_acc, exist_ok=True)
            os.makedirs(path_classification_devacc, exist_ok=True)   

            

            caminho_arquivo_cl = os.path.join(MAIN_PATH, file_cl)
            data = pd.read_csv(caminho_arquivo_cl, encoding="utf-8", on_bad_lines="skip")

            devacc_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'Epochs': data['Epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'Nhid': data['Nhid']}
            acc_data = {'Modelo': data['Modelo'], 'Pooling': data['Pooling'], 'Epochs': data['Epochs'], 'out_vec_size': data['out_vec_size'], 'qtd_layers': data['qtd_layers'], 'Nhid': data['Nhid']}

            for task in columns_tasks_cl:
                devacc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('devacc', None))
                acc_data[task] = data[task].apply(lambda x: parse_dict_with_eval(x).get('acc', None))

            devacc_table = pd.DataFrame(devacc_data)
            acc_table = pd.DataFrame(acc_data)

            devacc_table = devacc_table[ordem_colunas_cl]
            acc_table = acc_table[ordem_colunas_cl]

            devacc_table['Avg'] = devacc_table[columns_tasks_cl].mean(axis=1)
            acc_table['Avg'] = acc_table[columns_tasks_cl].mean(axis=1)

            devacc_table.to_csv(os.path.join(path_classification_devacc, file_cl.split('.csv')[0] + '_devacc.csv'))
            acc_table.to_csv(os.path.join(path_classification_acc, file_cl.split('.csv')[0]) + '_acc.csv')

            shutil.move(caminho_arquivo_cl, experiment_path + "/" + file_cl) 

    elif task_type_args == "similarity":
        print('file' + "ok")

if __name__ == "__main__":
    main()