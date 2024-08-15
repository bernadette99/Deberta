from DiskVectorIndex import DiskVectorIndex
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import math

# Définition la clé API Cohere
os.environ['COHERE_API_KEY'] = "RwbHpQF9iH6VGmEhpS1vrkimGJMxF6sOVkhredbs"

# Chargement de l'index
index = DiskVectorIndex("Cohere/trec-rag-2024-index")

# Liste des modèles de reranking à utiliser
reranker_model_names = [
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'cross-encoder/stsb-roberta-large',
    'jinaai/jina-reranker-v2-base-multilingual',
    'cross-encoder/nli-deberta-v3-base'
]

# Chargement des modèles et tokenizers
def load_models(model_names):
    models = {}
    tokenizers = {}
    for name in model_names:
        tokenizers[name] = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        models[name] = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True)
        models[name].eval()
    return models, tokenizers

models, tokenizers = load_models(reranker_model_names)

# Initialisation de la fonction sigmoïde
sigmoid = torch.nn.Sigmoid()

# Fonction pour reranker les candidats en utilisant le modèle de rerank
def rerank(query, candidates, models, tokenizers):
    rerank_scores = {model_name: [] for model_name in models.keys()}
    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]
        for candidate in candidates:
            doc_text = candidate['doc']['segment']
            inputs = tokenizer(query, doc_text, return_tensors='pt', truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            logits = logits.to(torch.float32)
            probs = sigmoid(logits).cpu().numpy()
            score = probs[0][1] if probs.shape[1] == 2 else np.max(probs)
            rerank_scores[model_name].append(score)
    ranked_indices = {model_name: np.argsort(scores)[::-1] for model_name, scores in rerank_scores.items()}
    return ranked_indices, rerank_scores


# Fonction pour charger les données qrel
def load_qrel(qrel_file):
    relevance = {}
    with open(qrel_file, 'r') as file:
        for line in file:
            query_id, _, doc_id, relevance_level = line.strip().split()
            relevance_level = int(relevance_level)
            if query_id not in relevance:
                relevance[query_id] = {}
            relevance[query_id][doc_id] = relevance_level
    return relevance

# Fonction pour calculer la précision à k (Precision@k)
def precision_at_k(retrieved_docs, relevant_docs, query_id, k):
    relevant = relevant_docs.get(query_id, {})
    relevant_set = set(doc_id for doc_id, rel in relevant.items() if rel > 0)
    retrieved_relevant = 0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_set:
            retrieved_relevant += 1
    return retrieved_relevant / k

# Fonction pour calculer l'Average Precision (AP)
def average_precision(retrieved_docs, relevant_docs, query_id):
    relevant = relevant_docs.get(query_id, {})
    relevant_set = set(doc_id for doc_id, rel in relevant.items() if rel > 0)
    retrieved_relevant = 0
    sum_precision = 0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            retrieved_relevant += 1
            precision_at_i = retrieved_relevant / (i + 1)
            sum_precision += precision_at_i
    num_relevant = len(relevant_set)
    return sum_precision / num_relevant if num_relevant > 0 else 0

# Fonction pour calculer le Discounted Cumulative Gain (DCG) à k
def discounted_cumulative_gain_at_k(retrieved_docs, relevant_docs, query_id, k):
    relevant = relevant_docs.get(query_id, {})
    dcg = 0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        relevance = relevant.get(doc_id, 0)
        if relevance > 0:
            dcg += (2 ** relevance - 1) / math.log2(i + 2)  # i+2 pour éviter log2(1) = 0
    return dcg

# Fonction pour calculer l'Ideal Discounted Cumulative Gain (IDCG) à k
def ideal_discounted_cumulative_gain_at_k(retrieved_docs, relevant_docs, query_id, k):
    relevant = relevant_docs.get(query_id, {})
    sorted_relevant = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0
    for i, relevance in enumerate(sorted_relevant):
        if relevance > 0:
            idcg += (2 ** relevance - 1) / math.log2(i + 2)
    return idcg

# Fonction pour calculer le Normalized Discounted Cumulative Gain (NDCG)
def normalized_discounted_cumulative_gain(retrieved_docs, relevant_docs, query_id, k):
    dcg = discounted_cumulative_gain_at_k(retrieved_docs, relevant_docs, query_id, k)
    idcg = ideal_discounted_cumulative_gain_at_k(retrieved_docs, relevant_docs, query_id, k)
    return dcg / idcg if idcg > 0 else 0

# Fonction pour calculer la Mean Average Precision (MAP)
def mean_average_precision(retrieved_docs_by_query, relevant_docs):
    ap_scores = []
    for query_id, retrieved_docs in retrieved_docs_by_query.items():
        ap = average_precision(retrieved_docs, relevant_docs, query_id)
        ap_scores.append(ap)
    return np.mean(ap_scores) if ap_scores else 0

# Fonction pour calculer le Recall à k
def recall_at_k(retrieved_docs, relevant_docs, query_id, k):
    relevant = relevant_docs.get(query_id, {})
    relevant_set = set(doc_id for doc_id, rel in relevant.items() if rel > 0)
    retrieved_docs_at_k = set(retrieved_docs[:k])
    
    num_relevant_retrieved = sum(1 for doc_id in retrieved_docs_at_k if doc_id in relevant_set)
    num_relevant_total = len(relevant_set)
    
    return num_relevant_retrieved / num_relevant_total if num_relevant_total > 0 else 0


# Fonction pour évaluer les modèles
def evaluate_models(queries, index, models, tokenizers, relevant_docs):
    # Stocker les résultats des évaluations
    model_metrics = {model_name: {'precision@k': [], 'recall': [], 'ndcg@k': [], 'map': []} for model_name in models.keys()}

    for line in queries:
        query_id, query = line.strip().split('\t')
        
        print(f"Query ID: {query_id}")
        print(f"Query: {query}")
        print("=========")
        
        docs = index.search(query, top_k=10)
        ranked_indices, rerank_scores = rerank(query, docs, models, tokenizers)

        for model_name, indices in ranked_indices.items():
            ranked_docs = [docs[idx]['doc']['docid'] for idx in indices]
            relevant_docs_for_query = [doc_id for doc_id in relevant_docs.get(query_id, [])]


            print(f"\nResults for model: {model_name}")
            for rank, idx in enumerate(indices):
                doc = docs[idx]
                print(f"Rank {rank + 1}: Document ID {doc['doc']['docid']}, Score {rerank_scores[model_name][idx]}")
                #print(doc)
                print("=========")
                
            precision_k = precision_at_k(ranked_docs, relevant_docs, query_id, k=10)
            recall_k = recall_at_k(ranked_docs, relevant_docs, query_id, k=100)
            map = mean_average_precision({query_id: ranked_docs}, relevant_docs)
            ndcg = normalized_discounted_cumulative_gain(ranked_docs, relevant_docs, query_id, k=10)

            model_metrics[model_name]['precision@k'].append(precision_k)
            model_metrics[model_name]['recall'].append(recall_k)
            model_metrics[model_name]['ndcg@k'].append(ndcg)
            model_metrics[model_name]['map'].append(map)

    # Calcul des moyennes des métriques pour chaque modèle
    for model_name, metrics in model_metrics.items():
        precision_at_k_avg = np.mean(metrics['precision@k'])
        recall_avg = np.mean(metrics['recall'])
        ndcg_at_k_avg = np.mean(metrics['ndcg@k'])
        map_avg = np.mean(metrics['map'])
        
        print(f"Model: {model_name}")
        print(f"Average Precision@10: {precision_at_k_avg:.4f}")
        print(f"Average Recall: {recall_avg:.4f}")
        print(f"Average NDCG@10: {ndcg_at_k_avg:.4f}")
        print(f"Average MAP: {map_avg:.4f}")
        print("=========")

if __name__ == '__main__':
    # Chargement des données qrel
    qrel_file = '../donnees-msmarco/qrels.rag24.raggy-dev.txt'
    relevant_docs = load_qrel(qrel_file)
    
    # Lire les requêtes
    query_file = '../donnees-msmarco/topics.rag24.raggy-dev.txt'
    with open(query_file, 'r') as file:
        queries = file.readlines()

    evaluate_models(queries, index, models, tokenizers, relevant_docs)
