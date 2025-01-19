import senteval
from transformers import AutoTokenizer, AutoModelForMaskedLM, DebertaV2Model, DebertaV2Tokenizer, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AutoModel, DistilBertTokenizer, DistilBertModel
import torch
import argparse
import pandas as pd
import logging
import os

class SentenceEncoder:
    def __init__(self, model_name, pooling_strategy, device):
        self.device = device
        self.size_embedding = None
        self.pooling_strategy = pooling_strategy

        if model_name == 'bert-base' or  model_name == 'bert-large':
            if model_name == 'bert-base':
                self.name_model = 'google-bert/bert-base-uncased'
                self.qtd_layers = 12
            if model_name == 'bert-large':
                self.name_model = 'google-bert/bert-large-uncased'
                self.qtd_layers = 24
            self.tokenizer = BertTokenizer.from_pretrained(self.name_model)
            self.model = BertModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'roberta-base' or  model_name == 'roberta-large':
            if model_name == 'roberta-base':
                self.name_model = 'FacebookAI/roberta-base'
                self.qtd_layers = 12
            if model_name == 'roberta-large':
                self.name_model = 'FacebookAI/roberta-large'
                self.qtd_layers = 24
            self.tokenizer = RobertaTokenizer.from_pretrained(self.name_model)
            self.model = RobertaModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'deberta-base' or model_name == 'deberta-large':
            if model_name == 'deberta-base':
                self.name_model = 'microsoft/deberta-v3-base'
                self.qtd_layers = 12
            if model_name == 'deberta-large':
                self.name_model = 'microsoft/deberta-v3-large'
                self.qtd_layers = 24
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.name_model)
            self.model = DebertaV2Model.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'distilbert-base':
            self.name_model = 'distilbert/distilbert-base-uncased'
            self.qtd_layers = 12
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.name_model)
            self.model = DistilBertModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'modernbert-base' or model_name == 'modernbert-large':
            if model_name == 'modernbert-base':
                self.name_model = 'answerdotai/ModernBERT-base'
                self.qtd_layers = 22
            if model_name == 'modernbert-large':
                self.name_model = 'answerdotai/ModernBERT-large'
                self.qtd_layers = 28
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            self.model = AutoModelForMaskedLM.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'allmini6':
            self.name_model = 'sentence-transformers/all-MiniLM-L6-v2'
            self.qtd_layers = 6
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            self.model = AutoModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)
        
        if model_name == 'allmpnet':
            self.name_model = 'sentence-transformers/all-mpnet-base-v2'
            self.qtd_layers = 12
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            self.model = AutoModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'sentencet5-11b':
            self.name_model = 'sentence-transformers/sentence-t5-xxl'
            self.qtd_layers = 24
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            self.model = AutoModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

    def _encode(self, sentences, batch_size=1024): 
        # Pré-tokenizar todas as frases
        tokens = self.tokenizer(
            sentences, padding="longest", truncation=True, return_tensors="pt", max_length = self.model.config.max_position_embeddings
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
        self.size_embedding = embeddings[0].shape
        return torch.cat(all_embeddings, dim=0).to('cpu').numpy()
   
    def _mean_pooling_exclude_cls_sep(self, output, attention_mask):
 
        # Exclui o CLS removendo o primeiro token
        embeddings = output[:, 1:-1, :]  # Remove CLS (primeiro) e SEP (último)
        attention_mask = attention_mask[:, 1:-1]  # Remove CLS e SEP na máscara também

        # Expande a máscara para corresponder ao tamanho dos embeddings
        expanded_mask = attention_mask.unsqueeze(-1)  # (batch_size x seq_len-2 x 1)

        # Aplica a máscara para excluir padding
        masked_embeddings = embeddings * expanded_mask

        # Soma os embeddings válidos e calcula a média
        sum_embeddings = masked_embeddings.sum(dim=1)  # Soma ao longo da sequência (dim=1)
        valid_token_counts = expanded_mask.sum(dim=1)  # Soma a quantidade de tokens válidos
        mean_pooled_embeddings = sum_embeddings / valid_token_counts.clamp(min=1e-9)  # Evita divisão por zero

        return mean_pooled_embeddings
    
    def _sum_pooling_exclude_cls_sep(self, output, attention_mask):

        # Exclui o CLS removendo o primeiro token e exclui o SEP removendo o último token
        embeddings = output[:, 1:-1, :]  # Remove CLS (primeiro) e SEP (último)
        attention_mask = attention_mask[:, 1:-1]  # Remove CLS e SEP na máscara também

        # Expande a máscara para corresponder ao tamanho dos embeddings
        expanded_mask = attention_mask.unsqueeze(-1)  # (batch_size x seq_len-2 x 1)

        # Aplica a máscara para excluir padding
        masked_embeddings = embeddings * expanded_mask

        # Soma os embeddings válidos ao longo da sequência
        sum_pooled_embeddings = masked_embeddings.sum(dim=1)

        return sum_pooled_embeddings

    def _max_pooling_exclude_cls_sep(self, output, attention_mask):

        # Exclui o CLS removendo o primeiro token e exclui o SEP removendo o último token
        embeddings = output[:, 1:-1, :]  # Remove CLS (primeiro) e SEP (último)
        attention_mask = attention_mask[:, 1:-1]  # Remove CLS e SEP na máscara também

        # Expande a máscara para corresponder ao tamanho dos embeddings
        expanded_mask = attention_mask.unsqueeze(-1)  # (batch_size x seq_len-2 x 1)

        # Substitui tokens de padding por um valor muito pequeno (-inf) para ignorá-los
        masked_embeddings = embeddings.masked_fill(expanded_mask == 0, float('-inf'))

        # Seleciona o valor máximo ao longo da sequência
        max_pooled_embeddings = masked_embeddings.max(dim=1).values

        return max_pooled_embeddings

    def _preprocess_output(self, output, attention_mask, exclude_cls_sep=False):
        if exclude_cls_sep:
            # Remove CLS e SEP ajustando os índices
            output = output[:, 1:-1, :]  # Exclui CLS (primeiro) e SEP (último) token
            attention_mask = attention_mask[:, 1:-1]  # Ajusta a máscara para corresponder
        
        # Aplica a máscara para excluir tokens de PAD
        valid_tokens = attention_mask.unsqueeze(-1).expand(output.size())
        processed_output = output * valid_tokens  # Zera os embeddings dos tokens de PAD

        return processed_output, attention_mask
    
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

    def _get_pooling_result(self, hidden_state, attention_mask, name_pooling):

        cls_result = hidden_state[:, 0, :]
        avg_result = ((hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
        sum_result = (hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
        max_result = torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1).expand(hidden_state.size()).float() == 0, -1e9), dim=1)[0]

        match name_pooling:

            case "CLS+AVG+AVG-NS":
                return torch.cat((cls_result, avg_result, self._mean_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+AVG+SUM-NS":
                return torch.cat((cls_result, avg_result, self._sum_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+AVG+MAX-NS":
                return torch.cat((cls_result, avg_result, self._max_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
                        
            case "CLS+SUM+AVG-NS":
                return torch.cat((cls_result, sum_result, self._mean_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+SUM+SUM-NS":
                return torch.cat((cls_result, sum_result, self._sum_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+SUM+MAX-NS":
                return torch.cat((cls_result, sum_result, self._max_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+MAX+AVG-NS":
                return torch.cat((cls_result, max_result, self._mean_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+MAX+SUM-NS":
                return torch.cat((cls_result, max_result, self._sum_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+MAX+MAX-NS":
                return torch.cat((cls_result, max_result, self._max_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)

            case "CLS":
                return cls_result
            
            case "AVG":
                return avg_result
            
            case "SUM":
                return sum_result
            
            case "MAX":
                return max_result
            
            case "AVG-NS":
                return self._mean_pooling_exclude_cls_sep(hidden_state, attention_mask)
            
            case "SUM-NS":
                return self._sum_pooling_exclude_cls_sep(hidden_state, attention_mask)
            
            case "MAX-NS":
                return self._max_pooling_exclude_cls_sep(hidden_state, attention_mask)
                                    
            case "CLS+AVG":
                return torch.cat((cls_result, avg_result), dim=1)
            
            case "CLS+SUM":
                return torch.cat((cls_result, sum_result), dim=1)
            
            case "CLS+MAX":
                return torch.cat((cls_result, max_result), dim=1)
                        
            case "CLS+AVG-NS":
                return torch.cat((cls_result, self._mean_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+SUM-NS":
                return torch.cat((cls_result, self._sum_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+MAX-NS":
                return torch.cat((cls_result, self._max_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "DY3":
                output_no_pad, _ = self._preprocess_output(hidden_state, attention_mask, exclude_cls_sep=False)      
                return self._concat_n_means(output_no_pad, 3)
            
            case "DY3-NS":
                output_no_special, _ = self._preprocess_output(hidden_state, attention_mask, exclude_cls_sep=True)
                return self._concat_n_means(output_no_special , 3)
            
            case "DY5":
                output_no_pad, _ = self._preprocess_output(hidden_state, attention_mask, exclude_cls_sep=False)      
                return self._concat_n_means(output_no_pad, 5)
            
            case "DY5-NS":
                output_no_special, _ = self._preprocess_output(hidden_state, attention_mask, exclude_cls_sep=True)
                return self._concat_n_means(output_no_special , 5)   
                         
    def _apply_pooling(self, output, attention_mask):
        hidden_states = output.hidden_states

        if self.pooling_strategy.split("_")[-1].startswith("LYR"):
            layer_idx = int(self.pooling_strategy.split("_")[-1].split('-')[-1])   
            LYR_hidden =  hidden_states[layer_idx]            
            name_pooling = self.pooling_strategy.split("_")[0]

            return self._get_pooling_result(LYR_hidden, attention_mask, name_pooling)
        
        if self.pooling_strategy.split("_")[-1] == "SUML4L":
            SUML4L_hidden = torch.stack(hidden_states[-4:], dim=0).sum(dim=0)      
            name_pooling = self.pooling_strategy.split("_")[0]

            return self._get_pooling_result(SUML4L_hidden, attention_mask, name_pooling)
        
        if self.pooling_strategy.split("_")[-1] == "AVGL4L":
            AVGL4L_hidden = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)      
            name_pooling = self.pooling_strategy.split("_")[0]

            return self._get_pooling_result(AVGL4L_hidden, attention_mask, name_pooling)
        
        if self.pooling_strategy.split("_")[-1] == "SUMALL":
            SUMALL_hidden = torch.stack(hidden_states[1:], dim=0).sum(dim=0)          
            name_pooling = self.pooling_strategy.split("_")[0]

            return self._get_pooling_result(SUMALL_hidden, attention_mask, name_pooling)
        
        if self.pooling_strategy.split("_")[-1] == "AVGALL":
            AVGALL_hidden = torch.stack(hidden_states[1:], dim=0).mean(dim=0)            
            name_pooling = self.pooling_strategy.split("_")[0]

            return self._get_pooling_result(AVGALL_hidden, attention_mask, name_pooling)

    def _strategies_pooling_list (self):

        simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']
        simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS']
        two_tokens_poolings = ['CLS+AVG', 'CLS+SUM', 'CLS+MAX', 'CLS+AVG-NS', 'CLS+SUM-NS', 'CLS+MAX-NS']
        three_tokens_poolings = ['CLS+AVG+AVG-NS', 'CLS+AVG+SUM-NS', 'CLS+AVG+MAX-NS', 
                                'CLS+SUM+AVG-NS', 'CLS+SUM+SUM-NS', 'CLS+SUM+MAX-NS', 
                                'CLS+MAX+AVG-NS', 'CLS+MAX+SUM-NS', 'CLS+MAX+MAX-NS']
        dynamic_poolings = ['DY3', 'DY3-NS', 'DY5', 'DY5-NS']
    
        #pooling_prefixs = simple_poolings + simple_ns_poolings + two_tokens_poolings + three_tokens_poolings + dynamic_poolings
        pooling_prefixs = simple_poolings + simple_ns_poolings + two_tokens_poolings + three_tokens_poolings
        initial_layer = int(self.qtd_layers / 2)
        
        #ONE LAYER STRATEGIES
        one_layer_strategies = []
        for i in range(initial_layer, self.qtd_layers):
            for p in pooling_prefixs:
                one_layer_strategies.append(p + f"_LYR-{i+1}")

        #SUM LAST 4 LAYERS STRATEGIES
        sum_last_4_layers_strategies = []
        for p in pooling_prefixs:
            sum_last_4_layers_strategies.append(p + f"_SUML4L")   

        #AVG LAST 4 LAYERS STRATEGIES
        avg_last_4_layers_strategies = []
        for p in pooling_prefixs:
            avg_last_4_layers_strategies.append(p + f"_AVGL4L")  
        
        #SUM ALL LAYERS STRATEGIES
        sum_all_layers_strategies = []
        for p in pooling_prefixs:
            sum_all_layers_strategies.append(p + f"_SUMALL")
        
        #AVG ALL LAYERS STRATEGIES
        avg_all_layers_strategies = []
        for p in pooling_prefixs:
            avg_all_layers_strategies.append(p + f"_AVGALL") 

        pooling_strategies = one_layer_strategies + sum_last_4_layers_strategies + avg_last_4_layers_strategies + sum_all_layers_strategies + avg_all_layers_strategies
        
        return pooling_strategies

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def batcher(params, batch):
    sentences = [' '.join(sent) for sent in batch]
    return params['encoder']._encode(sentences)

def run_senteval(model_name, tasks, epochs, nhid_number):

    device = get_device()
    print(device)
    results = {}
    encoder = SentenceEncoder(model_name, '', device)
    pooling_strategies = encoder._strategies_pooling_list()
    for pooling in pooling_strategies:
        encoder.pooling_strategy = pooling
        print(f"Running: Model={encoder.name_model}, Pooling={encoder.pooling_strategy}")
        senteval_params = {
            'task_path': 'data',
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
        results[pooling]['out_vec_size'] = encoder.size_embedding
        results[pooling]['qtd_layers'] = encoder.qtd_layers
        print(encoder.size_embedding)

    return results

def classification_run(models_args, epochs_args, nhid_args, main_path):

    filename_cl = f'cl' + "_nhid_" + str(nhid_args) + '_models_' + '&'.join([st for st in models_args]) + "_epochs_" + str(epochs_args)
    path_created = main_path + '/' + filename_cl
    os.makedirs(path_created, exist_ok=True)

    logging.basicConfig(
        filename=path_created + '/' + filename_cl + '_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    classification_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    classification_results_data = []

    for model_name in models_args:
        print(f"\nExecuting Model: {model_name}")
        results = run_senteval(model_name, classification_tasks, epochs_args, nhid_args)
        for pooling, res in results.items():

            classification_results = [res.get(task, {}) for task in classification_tasks]
            
            classification_results_data.append({
                "Modelo": model_name,
                "Pooling": pooling,
                "Epochs": epochs_args,
                "out_vec_size": res.get('out_vec_size'),
                "qtd_layers": res.get('qtd_layers'),
                "Nhid": nhid_args,
                **{task: classification_results[i] for i, task in enumerate(classification_tasks)}
            })
            
        df1 = pd.DataFrame(classification_results_data)
        df1.to_csv(path_created + '/' + filename_cl + '_intermediate.csv', index=False)
            
    classification_df = pd.DataFrame(classification_results_data)
    classification_df.to_csv(path_created + '/' + filename_cl + '.csv', index=False)

def similarity_run(models_args, epochs_args, nhid_args, main_path):

    filename_si = f'si' + "_nhid_" + str(nhid_args) + '_models_' + '&'.join([st for st in models_args]) + "_epochs_" + str(epochs_args)
    path_created = main_path + '/' + filename_si
    os.makedirs(path_created, exist_ok=True)

    logging.basicConfig(
        filename=path_created + '/' + filename_si + '_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    similarity_results_data = []

    for model_name in models_args:
        print(f"\nExecuting Model: {model_name}")
        results = run_senteval(model_name, similarity_tasks, epochs_args, nhid_args)
        for pooling, res in results.items():

            similarity_results = [res.get(task, {}).get('all', 0) for task in similarity_tasks]
            
            similarity_results_data.append({
                "Modelo": model_name,
                "Pooling": pooling,
                "Epochs": epochs_args,
                "out_vec_size": res.get('out_vec_size'),
                "qtd_layers": res.get('qtd_layers'),
                "Nhid": nhid_args,
                **{task: similarity_results[i] for i, task in enumerate(similarity_tasks)}
            })
            
        df1 = pd.DataFrame(similarity_results_data)
        df1.to_csv(path_created + '/' + filename_si + '_intermediate.csv', index=False)
            
    similarity_df = pd.DataFrame(similarity_results_data)
    similarity_df.to_csv(path_created + '/' + filename_si +  '.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="SentEval Experiments")
    parser.add_argument("--task_type", type=str, required=True, help="Tipo de tarefa (classification ou similarity)")
    parser.add_argument("--models", type=str, required=True, help="Modelos separados por vírgula (sem espaços)")
    parser.add_argument("--epochs", type=int, required=True, help="Número máximo de épocas do classificador linear")
    parser.add_argument("--nhid", type=int, required=True, help="Numero de camadas ocultas (0 = Logistic Regression, 1 ou mais = MLP)")
    parser.add_argument("--initial_layer", type=int, help="Camada inicial para execução dos experimentos")
    args = parser.parse_args()

    models_args = args.models.split(",")        
    epochs_args = args.epochs 
    nhid_args = args.nhid
    initial_layer_args = args.initial_layer 
    task_type_args = args.task_type 

    main_path = 'tables_main_experiments'

    if task_type_args == "classification":
        classification_run(models_args, epochs_args, nhid_args, main_path)

    elif task_type_args == "similarity":
        similarity_run(models_args, epochs_args, nhid_args, main_path)

if __name__ == "__main__":
    main()