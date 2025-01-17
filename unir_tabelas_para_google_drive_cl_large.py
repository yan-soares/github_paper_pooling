import pandas as pd

# Caminhos dos arquivos
files = [
    "/home/yandellwsl/resultados_csvs/experiments_results/cl_nhid-0_bert-large_roberta-large_deberta-large_epochs-1/classification_acc/cl_nhid-0_bert-large_roberta-large_deberta-large_epochs-1_processado_acc.csv"
]
main_path = "./"

df_baseline = "baselines/large_cl.tsv"
df_baseline = pd.read_csv(df_baseline, sep="\t")
df_baseline['nhid'] = 'not defined'
df_baseline['out_emb_size'] = 1024

dataframes = [df_baseline] + [pd.read_csv(file) for file in files]
combined_df = pd.concat(dataframes, ignore_index=True)

ordem_colunas = ['Modelo', 'Pooling', 'out_emb_size', 'epochs', 'qtd_layers', 'nhid',  
                 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'Avg_without_SST5']
combined_df = combined_df[ordem_colunas]

combined_df.to_csv(main_path + "/resultados_cl_large_with_baseline.csv", index=False)

# Converter valores para o padrão brasileiro (vírgula como separador decimal)
combined_df = combined_df.map(
    lambda x: f"{x:.2f}".replace(".", ",") if isinstance(x, (float, int)) else x
)

combined_df.to_csv(main_path + "/resultados_cl_large_with_baseline_google_drive.csv", index=False, sep=";")