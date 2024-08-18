from DiskVectorIndex import DiskVectorIndex
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import math
from tqdm import tqdm
import re

# Définir la clé API Cohere
os.environ['COHERE_API_KEY'] = ""

# Charger l'index
index = DiskVectorIndex("Cohere/trec-rag-2024-index")

# doc = index.search("how to cook breaded pork schnitzel in oven", top_k=5)

# print(doc)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Liste des modèles de reranking à utiliser
reranker_model_names = [
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'cross-encoder/stsb-roberta-large',
    'jinaai/jina-reranker-v2-base-multilingual',
    'cross-encoder/nli-deberta-v3-base'
]

# Charger les modèles et tokenizers
def load_models(model_names):
    """
    Charge les modèles et tokenizers spécifiés.

    Args:
        model_names (list of str): Liste des noms des modèles à charger.

    Returns:
        tuple: Deux dictionnaires, un pour les modèles et un pour les tokenizers.
    """
    models = {}
    tokenizers = {}
    for name in model_names:
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            models[name] = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True)
            models[name].eval()
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {name}: {e}")
    return models, tokenizers

models, tokenizers = load_models(reranker_model_names)

# Initialisation de la fonction sigmoïde
sigmoid = torch.nn.Sigmoid()


def rerank(query, candidates, models, tokenizers):
    rerank_scores = {model_name: [] for model_name in models.keys()}
    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]
        model.to(device)  # Déplace le modèle sur le GPU
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
            # supprimer le cache et libérer la mémoire
            del inputs, outputs, logits, probs
            torch.cuda.empty_cache()
    ranked_indices = {model_name: np.argsort(scores)[::-1] for model_name, scores in rerank_scores.items()}
    return ranked_indices, rerank_scores



def write_results_to_file(results, filename):
    """
    Écrit les résultats de reranking dans un fichier.

    Args:
        results (dict): Dictionnaire des résultats de reranking par modèle et requête.
        filename (str): Nom du fichier de sortie.
    """
    try:
        with open(filename, 'w') as f:
            for model_name, queries in results.items():
                for query_id, docs in queries.items():
                    for rank, (doc_id, score) in enumerate(docs):
                        f.write(f"{query_id} Q0 {doc_id} {rank + 1} {score} Jonah_Bernadette\n")
    except Exception as e:
        print(f"Erreur lors de l'écriture dans le fichier: {e}")

def evaluate_with_trec_eval(qrel_file, results_file):
    """
    Évalue les résultats en utilisant TREC eval.

    Args:
        qrel_file (str): Chemin du fichier QREL.
        results_file (str): Chemin du fichier contenant les résultats à évaluer.
    """
    try:
        os.system(f"trec_eval -m all_trec {qrel_file} {results_file}")
    except Exception as e:
        print(f"Erreur lors de l'évaluation avec TREC eval: {e}")


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def get_sanitized_results_filename(model_name):
    return f"{sanitize_filename(model_name)}_results.txt"



if __name__ == '__main__':
    
    qrel_file = '../donnees-msmarco/qrels.rag24.raggy-dev.txt'
    
    query_file = '../donnees-msmarco/topics.rag24.raggy-dev.txt'
    try:
        with open(query_file, 'r') as file:
            queries = file.readlines()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier de requêtes: {e}")
        queries = []

    # Préparation des résultats pour chaque modèle
    all_results = {model_name: {} for model_name in reranker_model_names}

   
    for line in tqdm(queries, desc="Processing Queries", unit="query"):
        query_id, query = line.strip().split('\t')
        #print(f"Processing query: {query_id}")

        # Recherche initiale
        docs = index.search(query, top_k=100)

        
        if not docs:
            print(f"No documents found for query: {query_id}")
            continue

        # Rerank les résultats avec tous les modèles
        ranked_indices, rerank_scores = rerank(query, docs, models, tokenizers)
        print("Reranking done")

        for model_name, indices in ranked_indices.items():
            # Préparation des résultats pour le modèle en cours
            ranked_docs = [(docs[idx]['doc']['docid'], rerank_scores[model_name][idx]) for idx in indices]
            all_results[model_name][query_id] = ranked_docs

    # Écrire les résultats dans un fichier
    results_file = 'results.txt'
    write_results_to_file(all_results, results_file)

    # Évaluation des résultats pour chaque modèle
    for model_name in reranker_model_names:
        model_results_file = get_sanitized_results_filename(model_name)
        write_results_to_file({model_name: all_results[model_name]}, model_results_file)
        print(f"\nEvaluation for model: {model_name}")
        evaluate_with_trec_eval(qrel_file, model_results_file)
