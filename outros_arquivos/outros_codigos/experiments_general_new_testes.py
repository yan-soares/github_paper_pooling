import senteval
from transformers import AutoModel, AutoTokenizer, DebertaV2Model, DebertaV2Tokenizer
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
        if model_name.startswith('microsoft/deberta'):
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
            self.model = DebertaV2Model.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.pooling_strategy = pooling_strategy

    def encode(self, sentences, batch_size=1024):
        # Pré-tokenizar todas as frases
        tokens = self.tokenizer(
            sentences, padding="longest", truncation=True, return_tensors="pt", max_length=1024
        )

        tokens = {key: val.to(self.device) for key, val in tokens.items()}  # Move os tokens para o dispositivo

        # Dividir os tokens em lotes
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_tokens = {key: val[i:i+batch_size] for key, val in tokens.items()}
            with torch.no_grad():
                output = self.model(**batch_tokens)
                embeddings = self._apply_pooling(output, batch_tokens['attention_mask'])
                all_embeddings.append(embeddings)
        #return torch.cat(all_embeddings, dim=0).cpu().numpy()
        return torch.cat(all_embeddings, dim=0).to('cpu').numpy()
   
    def _mean_pooling_exclude_cls_sep(self, output, attention_mask):
        # Garante que os tensores estejam na GPU
        attention_mask = attention_mask.to(self.device)
        output = output.to(self.device)

        # Exclui o CLS removendo o primeiro token
        embeddings = output[:, 1:, :]
        attention_mask = attention_mask[:, 1:]
        
        # Identifica o índice do último token válido (antes do padding)
        valid_lengths = attention_mask.sum(-1)  # Soma para encontrar o comprimento válido
        batch_size = output.size(0)
        
        # Cria uma máscara para excluir o SEP
        for i in range(batch_size):
            valid_lengths[i] -= 1  # Subtrai 1 para ignorar o SEP na última posição válida
            attention_mask[i, valid_lengths[i].item()] = 0  # Marca o SEP como inválido na máscara

        # Aplica a atenção ajustada
        expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.size())
        masked_embeddings = embeddings * expanded_mask

        # Calcula a média excluindo PAD, CLS e SEP
        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_token_counts = expanded_mask.sum(dim=1)
        mean_pooled_embeddings = sum_embeddings / valid_token_counts.clamp(min=1e-9)

        return mean_pooled_embeddings
    
    def _mean_pooling(self, output, attention_mask):
        # Garante que os tensores estejam na GPU
        attention_mask = attention_mask.to(self.device)
        output = output.to(self.device)
        return ((output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        #return ((output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
    
    def preprocess_output(self, output, attention_mask, exclude_cls_sep=False):
        # Garante que os tensores estejam na GPU
        attention_mask = attention_mask.to(self.device)
        output = output.to(self.device)

        if exclude_cls_sep:
            # Remove CLS e SEP ajustando os índices
            output = output[:, 1:-1, :]  # Exclui CLS (primeiro) e SEP (último) token
            attention_mask = attention_mask[:, 1:-1]  # Ajusta a máscara para corresponder
        
        # Aplica a máscara para excluir tokens de PAD
        valid_tokens = attention_mask.unsqueeze(-1).expand(output.size())
        processed_output = output * valid_tokens  # Zera os embeddings dos tokens de PAD

        return processed_output, attention_mask
    
    def _concat_n_means(self, output, n):
        # Garante que os tensores estejam na GPU
        output = output.to(self.device)

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
        if self.pooling_strategy.startswith("_"):     
            layer_idx = int(self.pooling_strategy.split("-")[-1])    
            cls_emb = hidden_states[layer_idx][:, 0, :]
            mean_emb = self._mean_pooling(hidden_states[layer_idx], attention_mask)
            mean_no_special_emb = self._mean_pooling_exclude_cls_sep(hidden_states[layer_idx], attention_mask)

            name_pooling = self.pooling_strategy.split("-")[0]
            if name_pooling == "_cls":
                return cls_emb
                        
            if name_pooling == "_mean":
                return mean_emb
            
            if name_pooling == "_mean_exclude_cls_sep":
                return mean_no_special_emb
            
            if name_pooling == "_concat_cls_with_mean":
                return torch.cat((cls_emb, mean_emb), dim=1)
            
            if name_pooling == "_concat_cls_with_mean_exclude_cls_sep":
                return torch.cat((cls_emb, mean_no_special_emb), dim=1)
            
            if name_pooling == "_concat_mean_3_tokens":
                output_no_pad, _ = self.preprocess_output(hidden_states[layer_idx], attention_mask, exclude_cls_sep=False)      
                return self._concat_n_means(output_no_pad, 3)
            
            if name_pooling == "_concat_mean_3_tokens_exclude_cls_and_sep":
                output_no_special, _ = self.preprocess_output(hidden_states[layer_idx], attention_mask, exclude_cls_sep=True)
                return self._concat_n_means(output_no_special , 3)
            
            if name_pooling == "_concat_mean_5_tokens":
                output_no_pad, _ = self.preprocess_output(hidden_states[layer_idx], attention_mask, exclude_cls_sep=False)      
                return self._concat_n_means(output_no_pad, 5)
                        
            if name_pooling == "_concat_mean_5_tokens_exclude_cls_and_sep":
                output_no_special, _ = self.preprocess_output(hidden_states[layer_idx], attention_mask, exclude_cls_sep=True)
                return self._concat_n_means(output_no_special , 5)

        ### ESTRATEGIAS DE POOLING PARA A SOMA DAS 4 ULTIMAS CAMADAS
        if self.pooling_strategy.endswith("last4"):            
            accumulated_last_4_layers = torch.stack(hidden_states[-4:], dim=0).sum(dim=0)      
            cls_emb_last_4_layers = accumulated_last_4_layers[:, 0, :]
            mean_emb_last_4_layers = self._mean_pooling(accumulated_last_4_layers, attention_mask)
            mean_no_special_emb_last_4_layers = self._mean_pooling_exclude_cls_sep(accumulated_last_4_layers, attention_mask)

            if self.pooling_strategy == "cls_last4":
                return cls_emb_last_4_layers
            
            if self.pooling_strategy == "mean_last4":
                return mean_emb_last_4_layers
            
            if self.pooling_strategy == "mean_exclude_cls_sep_last4":
                return mean_no_special_emb_last_4_layers
            
            if self.pooling_strategy == "concat_cls_with_mean_last4":
                return torch.cat((cls_emb_last_4_layers, mean_emb_last_4_layers), dim=1)
            
            if self.pooling_strategy == "concat_cls_with_mean_exclude_cls_sep_last4":
                return torch.cat((cls_emb_last_4_layers, mean_no_special_emb_last_4_layers), dim=1)
            
            if self.pooling_strategy == "concat_mean_3_tokens_last4":
                output_no_pad, _ = self.preprocess_output(accumulated_last_4_layers, attention_mask, exclude_cls_sep=False)      
                return self._concat_n_means(output_no_pad, 3)
            
            if self.pooling_strategy == "concat_mean_3_tokens_exclude_cls_and_sep_last4":
                output_no_special, _ = self.preprocess_output(accumulated_last_4_layers, attention_mask, exclude_cls_sep=True)
                return self._concat_n_means(output_no_special , 3)
            
            if self.pooling_strategy == "concat_mean_5_tokens_last4":
                output_no_pad, _ = self.preprocess_output(accumulated_last_4_layers, attention_mask, exclude_cls_sep=False)      
                return self._concat_n_means(output_no_pad, 5)
                        
            if self.pooling_strategy == "concat_mean_5_tokens_exclude_cls_and_sep_last4":
                output_no_special, _ = self.preprocess_output(accumulated_last_4_layers, attention_mask, exclude_cls_sep=True)
                return self._concat_n_means(output_no_special , 5)
        
        ### ESTRATEGIAS DE POOLING PARA A SOMA DE TODAS AS CAMADAS (MENOS A PRIMEIRA que nao é camada oculta)
        if self.pooling_strategy.endswith("sumall"):   
            accumulated_sumall_layers = torch.stack(hidden_states[1:], dim=0).sum(dim=0)      
            cls_emb_sumall_layers = accumulated_sumall_layers[:, 0, :]
            mean_emb_sumall_layers = self._mean_pooling(accumulated_sumall_layers, attention_mask)
            mean_no_special_emb_sumall_layers = self._mean_pooling_exclude_cls_sep(accumulated_sumall_layers, attention_mask)

            if self.pooling_strategy == "cls_sumall":
                return cls_emb_sumall_layers
            
            if self.pooling_strategy == "mean_sumall":
                return mean_emb_sumall_layers
            
            if self.pooling_strategy == "mean_exclude_cls_sep_sumall":
                return mean_no_special_emb_sumall_layers
            
            if self.pooling_strategy == "concat_cls_with_mean_sumall":
                return torch.cat((cls_emb_sumall_layers, mean_emb_sumall_layers), dim=1)
            
            if self.pooling_strategy == "concat_cls_with_mean_exclude_cls_sep_sumall":
                return torch.cat((cls_emb_sumall_layers, mean_no_special_emb_sumall_layers), dim=1)
            
            if self.pooling_strategy == "concat_mean_3_tokens_sumall":
                output_no_pad, _ = self.preprocess_output(accumulated_sumall_layers, attention_mask, exclude_cls_sep=False)      
                return self._concat_n_means(output_no_pad, 3)
            
            if self.pooling_strategy == "concat_mean_3_tokens_exclude_cls_and_sep_sumall":
                output_no_special, _ = self.preprocess_output(accumulated_sumall_layers, attention_mask, exclude_cls_sep=True)
                return self._concat_n_means(output_no_special , 3)
            
            if self.pooling_strategy == "concat_mean_5_tokens_sumall":
                output_no_pad, _ = self.preprocess_output(accumulated_sumall_layers, attention_mask, exclude_cls_sep=False)      
                return self._concat_n_means(output_no_pad, 5)
                        
            if self.pooling_strategy == "concat_mean_5_tokens_exclude_cls_and_sep_sumall":
                output_no_special, _ = self.preprocess_output(accumulated_sumall_layers, attention_mask, exclude_cls_sep=True)
                return self._concat_n_means(output_no_special , 5)
      
# Função para executar SentEval
def run_senteval(model_name, pooling_strategies, tasks, task_path, epochs, nhid_number):
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
                'nhid': nhid_number,
                'optim': 'adam',
                'batch_size': 512,
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
    parser.add_argument("--models", type=str, required=True, default="bert-base-uncased,roberta-base", help="Modelos HuggingFace separados por vírgulas")
    parser.add_argument("--pooling", type=str, default="", help="Pooling strategies")
    parser.add_argument("--epochs", type=int, default=1, help="Número máximo de épocas do classificador linear")
    parser.add_argument("--qtd_layers", type=int, default=12, help="quantidade de layers ocultas")
    parser.add_argument("--task_path", type=str, default="data", help="Caminho para os dados do SentEval")
    parser.add_argument("--nhid_number", type=int, default=0, help="Numero de camadas ocultas")
    parser.add_argument("--initial_layer", type=int, default=0, help="Numero de camadas ocultas")
    args = parser.parse_args()

    models = args.models.split(",")        

    more_strategies = []
    for i in range(args.initial_layer, args.qtd_layers):
        more_strategies.append(f"_cls-{i+1}")
        more_strategies.append(f"_mean-{i+1}")
        more_strategies.append(f"_mean_exclude_cls_sep-{i+1}")
        more_strategies.append(f"_concat_cls_with_mean-{i+1}")
        more_strategies.append(f"_concat_cls_with_mean_exclude_cls_sep-{i+1}")
        more_strategies.append(f"_concat_mean_3_tokens-{i+1}")
        more_strategies.append(f"_concat_mean_3_tokens_exclude_cls_and_sep-{i+1}")
        more_strategies.append(f"_concat_mean_5_tokens-{i+1}")
        more_strategies.append(f"_concat_mean_5_tokens_exclude_cls_and_sep-{i+1}")

    more_strategies += ["cls_last4", "mean_last4", "mean_exclude_cls_sep_last4", "concat_cls_with_mean_last4", "concat_cls_with_mean_exclude_cls_sep_last4", "concat_mean_3_tokens_last4", "concat_mean_3_tokens_exclude_cls_and_sep_last4", "concat_mean_5_tokens_last4", "concat_mean_5_tokens_exclude_cls_and_sep_last4", 
                        "cls_sumall", "mean_sumall", "mean_exclude_cls_sep_sumall", "concat_cls_with_mean_sumall", "concat_cls_with_mean_exclude_cls_sep_sumall", "concat_mean_3_tokens_sumall", "concat_mean_3_tokens_exclude_cls_and_sep_sumall", "concat_mean_5_tokens_sumall", "concat_mean_5_tokens_exclude_cls_and_sep_sumall"]

    pooling_strategies = more_strategies

    classification_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']
    classification_results_data = []

    # Configurar o logger
    logging.basicConfig(
        filename=f'log-results' + "_nhid" + str(args.nhid_number) + '_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '.txt',  # Nome do arquivo de saída
        level=logging.INFO,           # Nível do log
        format='%(asctime)s - %(levelname)s - %(message)s'  # Formato dos logs
    )

    for model_name in models:
        print(f"\nExecuting Model: {model_name}")
        results = run_senteval(model_name, pooling_strategies, classification_tasks, args.task_path, args.epochs, args.nhid_number)
        for pooling, res in results.items():
            classification_results = [res.get(task, {}) for task in classification_tasks]
            
            classification_results_data.append({
                "Modelo": model_name,
                "Pooling": pooling,
                "Nhid": args.nhid_number,
                **{task: classification_results[i] for i, task in enumerate(classification_tasks)}
            })
            
        # Salvar os resultados intermediarios como DataFrame
        df1 = pd.DataFrame(classification_results_data)
        df1.to_csv(f'classification-results' + "_nhid" + str(args.nhid_number) + '_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '_intermediate.csv', index=False)
            
    # Salvar os resultados finais como DataFrame
    classification_df = pd.DataFrame(classification_results_data)
    classification_df.to_csv(f'classification-results' + "_nhid" + str(args.nhid_number) + '_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '.csv', index=False)

if __name__ == "__main__":
    main()