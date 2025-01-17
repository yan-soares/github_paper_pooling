import senteval
from transformers import AutoModel, AutoTokenizer
import torch
import argparse
import pandas as pd
import logging

# Função para checar dispositivo
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Classe Encoder
class SentenceEncoder:
    def __init__(self, model_name, pooling_strategy, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.pooling_strategy = pooling_strategy

    def encode(self, sentences, batch_size=1024):
        # Pré-tokenizar todas as frases
        tokens = self.tokenizer(
            sentences, padding="longest", truncation=True, return_tensors="pt", max_length=512
        )
        # Dividir os tokens em lotes
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_tokens = {key: val[i:i+batch_size].to(self.device) for key, val in tokens.items()}
            with torch.no_grad():
                output = self.model(**batch_tokens)
                embeddings = self._apply_pooling(output, batch_tokens['attention_mask'])
                all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0).cpu().numpy()
   
    def _mean_no_cls_sep(self, output, attention_mask):
        token_embeddings = output[:, 1:-1, :]  # Exclui CLS e SEP
        input_mask_expanded = attention_mask[:, 1:-1].unsqueeze(-1).expand(token_embeddings.size())
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)
    
    def _concat_n_means(self, output, n):
        token_embeddings = output
        seq_len = token_embeddings.size(1)
        step = seq_len // n
        means = []
        for i in range(n):
            start, end = i * step, (i + 1) * step
            if i == n - 1:  # Garante que o último grupo pega os tokens restantes
                end = seq_len
            mean_segment = token_embeddings[:, start:end, :].mean(dim=1)
            means.append(mean_segment)
        return torch.cat(means, dim=1)

    def _apply_pooling(self, output, attention_mask):
        hidden_states = output.hidden_states
       
        ###ESTRATEGIAS DE POOLING PARA CAMADAS ÚNICAS
        if self.pooling_strategy.startswith("_cls_layer_"):
            layer_idx = int(self.pooling_strategy.split("_")[-1])
            cls_emb = hidden_states[layer_idx][:, 0, :]
            return cls_emb
        
        if self.pooling_strategy.startswith("_mean_pooling_"):
            layer_idx = int(self.pooling_strategy.split("_")[-1])
            mean_emb = hidden_states[layer_idx].mean(dim=1)
            return mean_emb
        
        if self.pooling_strategy.startswith("_mean_pooling_no_cls_and_sep_"):
            layer_idx = int(self.pooling_strategy.split("_")[-1])
            mean_no_special_emb = self._mean_no_cls_sep(hidden_states[layer_idx], attention_mask)
            return mean_no_special_emb
        
        if self.pooling_strategy.startswith("_concat_cls_layer_with_mean_pooling_"):
            layer_idx = int(self.pooling_strategy.split("_")[-1])
            cls_emb = hidden_states[layer_idx][:, 0, :]
            mean_emb = hidden_states[layer_idx].mean(dim=1)
            return torch.cat((cls_emb, mean_emb), dim=1)
        
        if self.pooling_strategy.startswith("_concat_cls_layer_with_mean_pooling_no_cls_and_sep_"):
            layer_idx = int(self.pooling_strategy.split("_")[-1])
            cls_emb = hidden_states[layer_idx][:, 0, :]
            mean_no_special_emb = self._mean_no_cls_sep(hidden_states[layer_idx], attention_mask)
            return torch.cat((cls_emb, mean_no_special_emb), dim=1)
        
        if self.pooling_strategy.startswith("_concat_3_tokens_mean_pooling_"):
            layer_idx = int(self.pooling_strategy.split("_")[-1])
            return self._concat_n_means(hidden_states[layer_idx], 3)
        
        if self.pooling_strategy.startswith("_concat_5_tokens_mean_pooling_"):
            layer_idx = int(self.pooling_strategy.split("_")[-1])
            return self._concat_n_means(hidden_states[layer_idx], 5)
        
        if self.pooling_strategy.startswith("_concat_3_tokens_mean_pooling_no_cls_and_sep_"):
            layer_idx = int(self.pooling_strategy.split("_")[-1])
            return self._concat_n_means(hidden_states[layer_idx][:, 1:-1, :] , 3)
        
        if self.pooling_strategy.startswith("_concat_5_tokens_mean_pooling_no_cls_and_sep_"):
            layer_idx = int(self.pooling_strategy.split("_")[-1])
            return self._concat_n_means(hidden_states[layer_idx][:, 1:-1, :] , 5)

        ### ESTRATEGIAS DE POOLING PARA A SOMA DAS 4 ULTIMAS CAMADAS
        last_4_layers = hidden_states[-4:]  # Pega as últimas 4 camadas
        accumulated_last_4_layers = torch.stack(last_4_layers, dim=0).sum(dim=0)  # Soma das 4 camadas        
        cls_emb_last_4_layers = accumulated_last_4_layers[:, 0, :]  # Embedding do token CLS
        mean_emb_last_4_layers = accumulated_last_4_layers.mean(dim=1)  # Média de todos os tokens
        mean_no_special_emb_last_4_layers = self._mean_no_cls_sep(accumulated_last_4_layers, attention_mask)  # Média sem CLS e SEP

        if self.pooling_strategy == "cls_accumulated_last_4_layers":
            return cls_emb_last_4_layers
        
        if self.pooling_strategy == "mean_pooling_accumulated_last_4_layers":
            return mean_emb_last_4_layers
        
        if self.pooling_strategy == "mean_pooling_no_cls_and_sep_accumulated_last_4_layers":
            return mean_no_special_emb_last_4_layers
        
        if self.pooling_strategy == "concat_cls_with_mean_pooling_accumulated_last_4_layers":
            return torch.cat((cls_emb_last_4_layers, mean_emb_last_4_layers), dim=1)
        
        if self.pooling_strategy == "concat_cls_with_mean_pooling_no_cls_and_sep_accumulated_last_4_layers":
            return torch.cat((cls_emb_last_4_layers, mean_no_special_emb_last_4_layers), dim=1)
        
        if self.pooling_strategy == "concat_3_tokens_mean_pooling_accumulated_last_4_layers":
            return self._concat_n_means(accumulated_last_4_layers, 3)
        
        if self.pooling_strategy == "concat_5_tokens_mean_pooling_accumulated_last_4_layers":
            return self._concat_n_means(accumulated_last_4_layers, 5)
        
        if self.pooling_strategy == "concat_3_tokens_mean_pooling_no_cls_and_sep_accumulated_last_4_layers":
            return self._concat_n_means(accumulated_last_4_layers[:, 1:-1, :], 3)
        
        if self.pooling_strategy == "concat_5_tokens_mean_pooling_no_cls_and_sep_accumulated_last_4_layers":
            return self._concat_n_means(accumulated_last_4_layers[:, 1:-1, :], 5)

        ### ESTRATEGIAS DE POOLING PARA A SOMA DE TODAS AS CAMADAS (MENOS A PRIMEIRA)
        sumall_layers = hidden_states[1:]  # Pega as últimas 4 camadas
        accumulated_sumall_layers = torch.stack(sumall_layers, dim=0).sum(dim=0)  # Soma das 4 camadas        
        cls_emb_sumall_layers = accumulated_sumall_layers[:, 0, :]  # Embedding do token CLS
        mean_emb_sumall_layers = accumulated_sumall_layers.mean(dim=1)  # Média de todos os tokens
        mean_no_special_emb_sumall_layers = self._mean_no_cls_sep(accumulated_sumall_layers, attention_mask)  # Média sem CLS e SEP

        if self.pooling_strategy == "cls_accumulated_sumall_layers":
            return cls_emb_sumall_layers
        
        if self.pooling_strategy == "mean_pooling_accumulated_sumall_layers":
            return mean_emb_sumall_layers
        
        if self.pooling_strategy == "mean_pooling_no_cls_and_sep_accumulated_sumall_layers":
            return mean_no_special_emb_sumall_layers
        
        if self.pooling_strategy == "concat_cls_with_mean_pooling_accumulated_sumall_layers":
            return torch.cat((cls_emb_sumall_layers, mean_emb_sumall_layers), dim=1)
        
        if self.pooling_strategy == "concat_cls_with_mean_pooling_no_cls_and_sep_accumulated_sumall_layers":
            return torch.cat((cls_emb_sumall_layers, mean_no_special_emb_sumall_layers), dim=1)
        
        if self.pooling_strategy == "concat_3_tokens_mean_pooling_accumulated_sumall_layers":
            return self._concat_n_means(accumulated_sumall_layers, 3)
        
        if self.pooling_strategy == "concat_5_tokens_mean_pooling_accumulated_sumall_layers":
            return self._concat_n_means(accumulated_sumall_layers, 5)
        
        if self.pooling_strategy == "concat_3_tokens_mean_pooling_no_cls_and_sep_accumulated_sumall_layers":
            return self._concat_n_means(accumulated_sumall_layers[:, 1:-1, :], 3)
        
        if self.pooling_strategy == "concat_5_tokens_mean_pooling_no_cls_and_sep_accumulated_sumall_layers":
            return self._concat_n_means(accumulated_sumall_layers[:, 1:-1, :], 5)
      
# Função para executar SentEval
def run_senteval(model_name, pooling_strategies, tasks, task_path, epochs):
    device = get_device()
    print(device)
    results = {}
    for pooling in pooling_strategies:
        print(f"Running: Model={model_name}, Pooling={pooling}")
        encoder = SentenceEncoder(model_name, pooling, device)
        senteval_params = {
            'task_path': task_path,
            'usepytorch': True,
            'kfold': 5,
            'classifier': {
                'nhid': 0,
                'optim': 'adam',
                'batch_size': 256,
                'tenacity': 5,
                'epoch_size': epochs
            },
            'encoder': encoder
        }
        se = senteval.engine.SE(senteval_params, batcher)
        results[pooling] = se.eval(tasks)

    return results

# Função de batcher
def batcher(params, batch):
    sentences = [' '.join(sent) for sent in batch]
    return params['encoder'].encode(sentences)

# Main Function
def main():

    parser = argparse.ArgumentParser(description="SentEval Experiments")
    parser.add_argument("--models", type=str, required=True, help="Modelos HuggingFace separados por vírgulas")
    parser.add_argument("--pooling", type=str, default="", help="Pooling strategies")
    parser.add_argument("--epochs", type=int, default=1, help="Número máximo de épocas do classificador linear")
    parser.add_argument("--qtd_layers", type=int, default=12, help="quantidade de layers ocultas")
    parser.add_argument("--task_path", type=str, default="data", help="Caminho para os dados do SentEval")
    args = parser.parse_args()

    models = args.models.split(",")        

    more_strategies = []
    #for i in range(args.qtd_layers):
    for i in range(11,12):
        more_strategies.append(f"_cls_layer_{i+1}")
        more_strategies.append(f"_mean_pooling_{i+1}")
        more_strategies.append(f"_mean_pooling_no_cls_and_sep_{i+1}")
        more_strategies.append(f"_concat_cls_layer_with_mean_pooling_{i+1}")
        more_strategies.append(f"_concat_cls_layer_with_mean_pooling_no_cls_and_sep_{i+1}")
        more_strategies.append(f"_concat_3_tokens_mean_pooling_{i+1}")
        more_strategies.append(f"_concat_5_tokens_mean_pooling_{i+1}")
        more_strategies.append(f"_concat_3_tokens_mean_pooling_no_cls_and_sep_{i+1}")
        more_strategies.append(f"_concat_5_tokens_mean_pooling_no_cls_and_sep_{i+1}")

    more_strategies += ["cls_accumulated_last_4_layers", "mean_pooling_accumulated_last_4_layers", "mean_pooling_no_cls_and_sep_accumulated_last_4_layers", "concat_cls_with_mean_pooling_accumulated_last_4_layers", "concat_cls_with_mean_pooling_no_cls_and_sep_accumulated_last_4_layers", "concat_3_tokens_mean_pooling_accumulated_last_4_layers", "concat_5_tokens_mean_pooling_accumulated_last_4_layers", "concat_3_tokens_mean_pooling_no_cls_and_sep_accumulated_last_4_layers", "concat_5_tokens_mean_pooling_no_cls_and_sep_accumulated_last_4_layers", 
                        "cls_accumulated_sumall_layers", "mean_pooling_accumulated_sumall_layers", "mean_pooling_no_cls_and_sep_accumulated_sumall_layers", "concat_cls_with_mean_pooling_accumulated_sumall_layers", "concat_cls_with_mean_pooling_no_cls_and_sep_accumulated_sumall_layers", "concat_3_tokens_mean_pooling_accumulated_sumall_layers", "concat_5_tokens_mean_pooling_accumulated_sumall_layers", "concat_3_tokens_mean_pooling_no_cls_and_sep_accumulated_sumall_layers", "concat_5_tokens_mean_pooling_no_cls_and_sep_accumulated_sumall_layers"]

    pooling_strategies = more_strategies

    #classification_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']
    classification_tasks = ['CR']
    similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']

    classification_results_data = []
    similarity_results_data = []

    # Configurar o logger
    logging.basicConfig(
        filename=f'resultados/log_results_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '.txt',  # Nome do arquivo de saída
        level=logging.INFO,           # Nível do log
        format='%(asctime)s - %(levelname)s - %(message)s'  # Formato dos logs
    )

    for model_name in models:
        print(f"\nExecuting Model: {model_name}")
        results = run_senteval(model_name, pooling_strategies, classification_tasks + similarity_tasks, args.task_path, args.epochs)
        for pooling, res in results.items():
            classification_results = [res.get(task, {}) for task in classification_tasks]
            similarity_results = [res.get(task, {}).get('all', 0) for task in similarity_tasks[:5]] + [res.get(task, {}) for task in similarity_tasks[5:]]

            classification_results_data.append({
                "Modelo": model_name,
                "Pooling": pooling,
                **{task: classification_results[i] for i, task in enumerate(classification_tasks)}
            })
            
            similarity_results_data.append({
                "Modelo": model_name,
                "Pooling": pooling,
                **{task: similarity_results[i] for i, task in enumerate(similarity_tasks)}
            })

        # Salvar os resultados intermediarios como DataFrame
        df1 = pd.DataFrame(classification_results_data)
        df1.to_csv(f'resultados/classification_results_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '_intermediate.csv', index=False)
        df2 = pd.DataFrame(similarity_results_data)
        df2.to_csv(f'resultados/similarity_results_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '_intermediate.csv', index=False)    
            
    # Salvar os resultados finais como DataFrame
    classification_df = pd.DataFrame(classification_results_data)
    similarity_df = pd.DataFrame(similarity_results_data)

    classification_df.to_csv(f'resultados/classification_results_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '.csv', index=False)
    similarity_df.to_csv(f'resultados/similarity_results_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '.csv', index=False)

    print("\nResultados de Classificação:")
    print(classification_df)

    print("\nResultados de Similaridade:")
    print(similarity_df)

if __name__ == "__main__":
    main()
