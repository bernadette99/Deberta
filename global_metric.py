from DiskVectorIndex import DiskVectorIndex
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import math
from tqdm import tqdm  # Importer tqdm pour la barre de progression

# Définition la clé API Cohere
os.environ['COHERE_API_KEY'] = ""

# Chargement de l'index
index = DiskVectorIndex("Cohere/trec-rag-2024-index")

# Liste des modèles de reranking à utiliser
reranker_model_names = [
    'cross-encoder/ms-marco-MiniLM-L-12-v2',
    'cross-encoder/qnli-electra-base',
    'jinaai/jina-reranker-v2-base-multilingual',
    'BAAI/bge-reranker-v2-m3'
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
def rerank(query, candidates, models, tokenizers, device):
    rerank_scores = {model_name: [] for model_name in models.keys()}
    for model_name, model in models.items():
        model.to(device)  # Déplace le modèle sur le GPU
        tokenizer = tokenizers[model_name]
        for candidate in candidates:
            doc_text = candidate['doc']['segment']
            inputs = tokenizer(query, doc_text, return_tensors='pt', truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            logits = logits.to(torch.float32)
            probs = sigmoid(logits).cpu().numpy()
            score = probs[0][1] if probs.shape[1] == 2 else np.max(probs)
            rerank_scores[model_name].append(score)
            # Clear cache and free memory
            del inputs, outputs, logits, probs
            torch.cuda.empty_cache()
    ranked_indices = {model_name: np.argsort(scores)[::-1] for model_name, scores in rerank_scores.items()}
    return ranked_indices, rerank_scores

# Fonction pour charger les données qrel
def load_qrel(qrel_file):
    
    """
    Charge les données de pertinence depuis un fichier QREL.

    Args:
        qrel_file (str): Chemin du fichier QREL.

    Returns:
        dict: Dictionnaire des niveaux de pertinence par identifiant de requête et document.
    """
    relevance = {}
    try:
        with open(qrel_file, 'r') as file:
            for line in file:
                query_id, _, doc_id, relevance_level = line.strip().split()
                relevance_level = int(relevance_level)
                if query_id not in relevance:
                    relevance[query_id] = {}
                relevance[query_id][doc_id] = relevance_level
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier QREL: {e}")
    return relevance

# Fonction pour calculer la précision à k (Precision@k)
def precision_at_k(retrieved_docs, relevant_docs, query_id, k):
    #print("retrieved_docs", retrieved_docs)
    # print("\nrelevant_docs\n", relevant_docs)
    # print("#######################################################################################################")
    
    relevant = relevant_docs.get(query_id, {})
    relevant_set = set(doc_id for doc_id, rel in relevant.items() if rel > 0)
    retrieved_relevant = 0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_set:
            retrieved_relevant += 1
    return retrieved_relevant / k if k > 0 else 0

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
    model_metrics = {model_name: {'precision@k': [], 'recall@k': [], 'ndcg@k': [], 'map': []} for model_name in models.keys()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Définir le périphérique
    for line in tqdm(queries, desc="Processing Queries", unit="query"):
        query_id, query = line.strip().split('\t')
        
        docs = index.search(query, top_k=100)
        if not docs:
            continue  # Passer à la requête suivante si aucun document n'est retourné
        
        ranked_indices, rerank_scores = rerank(query, docs, models, tokenizers, device)

        for model_name, indices in ranked_indices.items():
            ranked_docs = [docs[idx]['doc']['docid'].split('#')[0] for idx in indices]  
            relevant_docs_for_query = [doc_id for doc_id in relevant_docs.get(query_id, [])]
            
            #print("ranked_docs", ranked_docs)
            

            precision_k = precision_at_k(ranked_docs, relevant_docs, query_id, k=10)
            recall_k = recall_at_k(ranked_docs, relevant_docs, query_id, k=100)
            map = mean_average_precision({query_id: ranked_docs}, relevant_docs)
            ndcg = normalized_discounted_cumulative_gain(ranked_docs, relevant_docs, query_id, k=10)
            
            # print(f"Model: {model_name}")
            # print(f"Average Precision@10: {precision_k:.4f}")
            # print(f"Average Recall: {recall_k:.4f}")
            # print(f"Average NDCG@10: {ndcg:.4f}")
            # print(f"Average MAP: {map:.4f}")
            # print("=========")
            

            model_metrics[model_name]['precision@k'].append(precision_k)
            model_metrics[model_name]['recall@k'].append(recall_k)
            model_metrics[model_name]['ndcg@k'].append(ndcg)
            model_metrics[model_name]['map'].append(map)

    # Calcul des moyennes des métriques pour chaque modèle
    for model_name, metrics in model_metrics.items():
        precision_at_k_avg = np.mean(metrics['precision@k'])
        recall_avg = np.mean(metrics['recall@k'])
        ndcg_at_k_avg = np.mean(metrics['ndcg@k'])
        map_avg = np.mean(metrics['map'])
        
        print(f"Model: {model_name}")
        print(f"Average Precision@10: {precision_at_k_avg:.4f}")
        print(f"Average Recall@100: {recall_avg:.4f}")
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
