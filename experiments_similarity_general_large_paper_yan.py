import senteval
from transformers import AutoTokenizer, AutoModelForMaskedLM, DebertaV2Model, DebertaV2Tokenizer, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
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
        self.size_embedding = None

        if model_name == 'bert-large':
            self.name_model = 'google-bert/bert-large-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(self.name_model)
            self.model = BertModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'roberta-large':
            self.name_model = 'FacebookAI/roberta-large'
            self.tokenizer = RobertaTokenizer.from_pretrained(self.name_model)
            self.model = RobertaModel.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'deberta-large':
            self.name_model = 'microsoft/deberta-v3-large'
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.name_model)
            self.model = DebertaV2Model.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        if model_name == 'modernbert-large':
            self.name_model = 'answerdotai/ModernBERT-large'
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_model)
            self.model = AutoModelForMaskedLM.from_pretrained(self.name_model, output_hidden_states=True).to(self.device)

        self.pooling_strategy = pooling_strategy

    def encode(self, sentences, batch_size=1024): 

        if self.name_model == 'google-bert/bert-large-uncased' or self.name_model == 'FacebookAI/roberta-large' or self.name_model == 'microsoft/deberta-v3-large':
            max_length_input = 1024
        if self.name_model == 'answerdotai/ModernBERT-large':
            max_length_input = 8192

        # Pré-tokenizar todas as frases
        tokens = self.tokenizer(
            sentences, padding="longest", truncation=True, return_tensors="pt", max_length=max_length_input
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
   
    def mean_pooling_exclude_cls_sep(self, output, attention_mask):
 
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
    
    def sum_pooling_exclude_cls_sep(self, output, attention_mask):

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

    def max_pooling_exclude_cls_sep(self, output, attention_mask):

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

    def preprocess_output(self, output, attention_mask, exclude_cls_sep=False):
        if exclude_cls_sep:
            # Remove CLS e SEP ajustando os índices
            output = output[:, 1:-1, :]  # Exclui CLS (primeiro) e SEP (último) token
            attention_mask = attention_mask[:, 1:-1]  # Ajusta a máscara para corresponder
        
        # Aplica a máscara para excluir tokens de PAD
        valid_tokens = attention_mask.unsqueeze(-1).expand(output.size())
        processed_output = output * valid_tokens  # Zera os embeddings dos tokens de PAD

        return processed_output, attention_mask
    
    def concat_n_means(self, output, n):
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

    def get_pooling_result(self, hidden_state, attention_mask, name_pooling):

        cls_result = hidden_state[:, 0, :]
        avg_result = ((hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9))
        sum_result = (hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
        max_result = torch.max(hidden_state.masked_fill(attention_mask.unsqueeze(-1).expand(hidden_state.size()).float() == 0, -1e9), dim=1)[0]
        #max_result = hidden_state.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf')).max(dim=1).values

        match name_pooling:
            case "CLS":
                return cls_result
            
            case "AVG":
                return avg_result
            
            case "SUM":
                return sum_result
            
            case "MAX":
                return max_result
            
            case "AVG-NS":
                return self.mean_pooling_exclude_cls_sep(hidden_state, attention_mask)
            
            case "SUM-NS":
                return self.sum_pooling_exclude_cls_sep(hidden_state, attention_mask)
            
            case "MAX-NS":
                return self.max_pooling_exclude_cls_sep(hidden_state, attention_mask)
                                    
            case "CLS+AVG":
                return torch.cat((cls_result, avg_result), dim=1)
            
            case "CLS+SUM":
                return torch.cat((cls_result, sum_result), dim=1)
            
            case "CLS+MAX":
                return torch.cat((cls_result, max_result), dim=1)
                        
            case "CLS+AVG-NS":
                return torch.cat((cls_result, self.mean_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+SUM-NS":
                return torch.cat((cls_result, self.sum_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "CLS+MAX-NS":
                return torch.cat((cls_result, self.max_pooling_exclude_cls_sep(hidden_state, attention_mask)), dim=1)
            
            case "DY3":
                output_no_pad, _ = self.preprocess_output(hidden_state, attention_mask, exclude_cls_sep=False)      
                return self.concat_n_means(output_no_pad, 3)
            
            case "DY3-NS":
                output_no_special, _ = self.preprocess_output(hidden_state, attention_mask, exclude_cls_sep=True)
                return self.concat_n_means(output_no_special , 3)
            
            case "DY5":
                output_no_pad, _ = self.preprocess_output(hidden_state, attention_mask, exclude_cls_sep=False)      
                return self.concat_n_means(output_no_pad, 5)
            
            case "DY5-NS":
                output_no_special, _ = self.preprocess_output(hidden_state, attention_mask, exclude_cls_sep=True)
                return self.concat_n_means(output_no_special , 5)                

    def _apply_pooling(self, output, attention_mask):
        hidden_states = output.hidden_states

        #if self.pooling_strategy == 'CLS-CL':
        #    return output.pooler_output

        if self.pooling_strategy.split("_")[-1].startswith("LYR"):
            layer_idx = int(self.pooling_strategy.split("_")[-1].split('-')[-1])   
            LYR_hidden =  hidden_states[layer_idx]            
            name_pooling = self.pooling_strategy.split("_")[0]

            return self.get_pooling_result(LYR_hidden, attention_mask, name_pooling)
        
        if self.pooling_strategy.split("_")[-1] == "SUML4L":
            SUML4L_hidden = torch.stack(hidden_states[-4:], dim=0).sum(dim=0)      
            name_pooling = self.pooling_strategy.split("_")[0]

            return self.get_pooling_result(SUML4L_hidden, attention_mask, name_pooling)
        
        if self.pooling_strategy.split("_")[-1] == "AVGL4L":
            AVGL4L_hidden = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)      
            name_pooling = self.pooling_strategy.split("_")[0]

            return self.get_pooling_result(AVGL4L_hidden, attention_mask, name_pooling)
        
        if self.pooling_strategy.split("_")[-1] == "SUMALL":
            SUMALL_hidden = torch.stack(hidden_states[1:], dim=0).sum(dim=0)          
            name_pooling = self.pooling_strategy.split("_")[0]

            return self.get_pooling_result(SUMALL_hidden, attention_mask, name_pooling)
        
        if self.pooling_strategy.split("_")[-1] == "AVGALL":
            AVGALL_hidden = torch.stack(hidden_states[1:], dim=0).mean(dim=0)            
            name_pooling = self.pooling_strategy.split("_")[0]

            return self.get_pooling_result(AVGALL_hidden, attention_mask, name_pooling)


def strategies_pooling_list (initial_layer, qtd_layers):
    pooling_prefixs = ['CLS_', 'AVG_', 'SUM_', 'MAX_', 'AVG-NS_', 'SUM-NS_', 'MAX-NS_', 
                       'CLS+AVG_', 'CLS+SUM_', 'CLS+MAX_', 'CLS+AVG-NS_', 'CLS+SUM-NS_',
                        'CLS+MAX-NS_', 'DY3_', 'DY3-NS_', 'DY5_', 'DY5-NS_']
    
    #ONE LAYER STRATEGIES
    one_layer_strategies = []
    for i in range(initial_layer, qtd_layers):
        for p in pooling_prefixs:
            one_layer_strategies.append(p + f"LYR-{i+1}")

    #SUM LAST 4 LAYERS STRATEGIES
    sum_last_4_layers_strategies = []
    for p in pooling_prefixs:
        sum_last_4_layers_strategies.append(p + f"SUML4L")   

    #AVG LAST 4 LAYERS STRATEGIES
    avg_last_4_layers_strategies = []
    for p in pooling_prefixs:
        avg_last_4_layers_strategies.append(p + f"AVGL4L")  
    
    #SUM ALL LAYERS STRATEGIES
    sum_all_layers_strategies = []
    for p in pooling_prefixs:
        sum_all_layers_strategies.append(p + f"SUMALL")  
    
    #AVG ALL LAYERS STRATEGIES
    avg_all_layers_strategies = []
    for p in pooling_prefixs:
        avg_all_layers_strategies.append(p + f"AVGALL") 

    pooling_strategies = one_layer_strategies + sum_last_4_layers_strategies + avg_last_4_layers_strategies + sum_all_layers_strategies + avg_all_layers_strategies
    #pooling_strategies = one_layer_strategies

    return pooling_strategies

# Função para executar SentEval
def run_senteval(model_name, initial_layer, tasks, task_path, epochs, nhid_number):

    if model_name == 'bert-large' or model_name == 'roberta-large' or model_name == 'deberta-large':
        qtd_layers = 24
        initial_layer = int(qtd_layers/2)
    if model_name == 'modernbert-large':
        qtd_layers = 28
        initial_layer = int(qtd_layers/2)
    #initial_layer = qtd_layers - 1
    pooling_strategies = strategies_pooling_list(initial_layer, qtd_layers)

    device = get_device()
    print(device)
    results = {}
    encoder = SentenceEncoder(model_name, '', device)
    for pooling in pooling_strategies:
        encoder.pooling_strategy = pooling
        print(f"Running: Model={encoder.name_model}, Pooling={encoder.pooling_strategy}")
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
        results[pooling]['OUT_EMB_SIZE'] = encoder.size_embedding
        print(encoder.size_embedding)

    return results

# Função de batcher
def batcher(params, batch):
    sentences = [' '.join(sent) for sent in batch]
    return params['encoder'].encode(sentences)

# Main Function
def main():
    parser = argparse.ArgumentParser(description="SentEval Experiments")
    parser.add_argument("--models", type=str, required=True, default="bert-large,roberta-large", help="Modelos HuggingFace separados por vírgulas")
    #parser.add_argument("--pooling", type=str, default="", help="Pooling strategies")
    parser.add_argument("--epochs", type=int, default=1, help="Número máximo de épocas do classificador linear")
    #parser.add_argument("--qtd_layers", type=int, default=12, help="quantidade de layers ocultas")
    parser.add_argument("--task_path", type=str, default="data", help="Caminho para os dados do SentEval")
    parser.add_argument("--nhid_number", type=int, default=0, help="Numero de camadas ocultas")
    parser.add_argument("--initial_layer", type=int, default=0, help="Numero de camadas ocultas")
    args = parser.parse_args()

    models = args.models.split(",")        

    #similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    similarity_results_data = []

    # Configurar o logger
    logging.basicConfig(
        filename=f'logsi' + "_nhid-" + str(args.nhid_number) + '_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs-" + str(args.epochs) + '.txt',  # Nome do arquivo de saída
        level=logging.INFO,           # Nível do log
        format='%(asctime)s - %(levelname)s - %(message)s'  # Formato dos logs
    )

    for model_name in models:
        print(f"\nExecuting Model: {model_name}")
        results = run_senteval(model_name, args.initial_layer, similarity_tasks, args.task_path, args.epochs, args.nhid_number)
        for pooling, res in results.items():
            #similarity_results = [res.get(task, {}).get('all', 0) for task in similarity_tasks[:-2]] + [res.get(task, {}) for task in similarity_tasks[-2:]]
            similarity_results = [res.get(task, {}).get('all', 0) for task in similarity_tasks]
            
            similarity_results_data.append({
                "Modelo": model_name,
                "Pooling": pooling,
                "out_emb_size": res.get('OUT_EMB_SIZE'),
                "Nhid": args.nhid_number,
                **{task: similarity_results[i] for i, task in enumerate(similarity_tasks)}
            })
            
        # Salvar os resultados intermediarios como DataFrame
        df1 = pd.DataFrame(similarity_results_data)
        df1.to_csv(f'si' + "_nhid-" + str(args.nhid_number) + '_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs-" + str(args.epochs) + '_intermediate.csv', index=False)
            
    # Salvar os resultados finais como DataFrame
    similarity_df = pd.DataFrame(similarity_results_data)
    similarity_df.to_csv(f'si' + "_nhid-" + str(args.nhid_number) + '_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs-" + str(args.epochs) + '.csv', index=False)

if __name__ == "__main__":
    main()