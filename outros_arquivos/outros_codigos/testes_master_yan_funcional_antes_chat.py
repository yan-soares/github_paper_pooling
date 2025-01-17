import senteval
from transformers import AutoModel, AutoTokenizer
import torch
import argparse
import pandas as pd

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
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            tokens = self.tokenizer(sentences[i:i+batch_size], padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            with torch.no_grad():
                output = self.model(**tokens)
                embeddings = self._apply_pooling(output, tokens['attention_mask'])
                all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0).cpu().numpy()

    def _apply_pooling(self, output, attention_mask):
        hidden_states = output.hidden_states
        
        strategies = {
            "cls": lambda: output.last_hidden_state[:, 0, :], #T1
            "mean": lambda: output.last_hidden_state.mean(dim=1), #T2
            "mean_no_cls_sep": lambda: self._mean_no_cls_sep(output, attention_mask),
            "mean_4_layers": lambda: torch.stack(hidden_states[-4:]).mean(0).mean(1),
            "mean_cls_mean_tokens": lambda: self._mean_cls_mean_tokens(output, attention_mask),
            "concat_cls_mean_tokens": lambda: self._concat_cls_mean_tokens(output, attention_mask),
            "mean_4_layers_cls": lambda: torch.stack(hidden_states[-4:]).mean(0)[:, 0, :],
            "mean_cls_mean_4_layers": lambda: self._mean_cls_mean_4_layers(output, attention_mask),
            "concat_cls_mean_4_layers": lambda: self._concat_cls_mean_4_layers(output, attention_mask),
            "concat_3_means": lambda: self._concat_n_means(output, attention_mask, 3),
            "concat_5_means": lambda: self._concat_n_means(output, attention_mask, 5),
        }

        for i, layer in enumerate(hidden_states[1:]):
            strategies[f"cls_layer_{i+1}"] = (lambda layer=layer: layer[:, 0, :])
            strategies[f"mean_layer_{i+1}"] = (lambda layer=layer: layer.mean(dim=1))

        return strategies[self.pooling_strategy]()

    def _mean_pooling(self, output, attention_mask):
        token_embeddings = output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _mean_no_cls_sep(self, output, attention_mask):
        token_embeddings = output.last_hidden_state[:, 1:-1, :]  # Exclui CLS e SEP
        input_mask_expanded = attention_mask[:, 1:-1].unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _mean_cls_mean_tokens(self, output, attention_mask):
        cls_emb = output.last_hidden_state[:, 0, :]
        mean_emb = self._mean_pooling(output, attention_mask)
        return (cls_emb + mean_emb) / 2

    def _concat_cls_mean_tokens(self, output, attention_mask):
        cls_emb = output.last_hidden_state[:, 0, :]
        mean_emb = self._mean_pooling(output, attention_mask)
        return torch.cat((cls_emb, mean_emb), dim=1)

    def _mean_cls_mean_4_layers(self, output, attention_mask):
        cls_emb = output.last_hidden_state[:, 0, :]
        mean_4_layers = torch.stack(output.hidden_states[-4:]).mean(0).mean(1)
        return (cls_emb + mean_4_layers) / 2

    def _concat_cls_mean_4_layers(self, output, attention_mask):
        cls_emb = output.last_hidden_state[:, 0, :]
        mean_4_layers = torch.stack(output.hidden_states[-4:]).mean(0).mean(1)
        return torch.cat((cls_emb, mean_4_layers), dim=1)

    def _concat_n_means(self, output, attention_mask, n):
        token_embeddings = output.last_hidden_state
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
    parser.add_argument("--pooling", type=str, default="cls,mean,mean_no_cls_sep,mean_4_layers,mean_cls_mean_tokens,concat_cls_mean_tokens,mean_4_layers_cls,mean_cls_mean_4_layers,concat_cls_mean_4_layers,concat_3_means,concat_5_means", help="Pooling strategies")
    parser.add_argument("--epochs", type=int, default=1, help="Número máximo de épocas do classificador linear")
    parser.add_argument("--qtd_layers", type=int, default=12, help="quantidade de layers ocultas")
    parser.add_argument("--task_path", type=str, default="data", help="Caminho para os dados do SentEval")
    args = parser.parse_args()

    models = args.models.split(",")

    more_strategies = []
    for i in range(args.qtd_layers):
        more_strategies.append(f"cls_layer_{i+1}")
        more_strategies.append(f"mean_layer_{i+1}")
    #more_strategies = []

    ###pooling_strategies = args.pooling.split(",") + more_strategies
    pooling_strategies = more_strategies

    #classification_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']
    classification_tasks = ['CR']
    similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']

    classification_results_data = []
    similarity_results_data = []

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
            df1.to_csv(f'classification_results_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '_intermediate.csv', index=False)
            df2 = pd.DataFrame(similarity_results_data)
            df2.to_csv(f'similarity_results_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '_intermediate.csv', index=False)
            '''
            print("\nResultados de Classificação Intermediarios:")
            print(df1)

            print("\nResultados de Similaridade intermediarios:")
            print(df2)
            '''

    # Salvar os resultados finais como DataFrame
    classification_df = pd.DataFrame(classification_results_data)
    similarity_df = pd.DataFrame(similarity_results_data)

    classification_df.to_csv(f'classification_results_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '.csv', index=False)
    similarity_df.to_csv(f'similarity_results_' + '_'.join([st.replace('/', '_') for st in models]) + "_epochs" + str(args.epochs) + "_qtdlayers" + str(args.qtd_layers) + '.csv', index=False)

    print("\nResultados de Classificação:")
    print(classification_df)

    print("\nResultados de Similaridade:")
    print(similarity_df)

if __name__ == "__main__":
    main()
