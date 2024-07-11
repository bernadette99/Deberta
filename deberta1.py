
import os
import json
import gzip
import logging
import torch
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from functools import partial
import uuid
from rank_bm25 import BM25Okapi
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialiser le client Qdrant
client = QdrantClient(host='localhost', port=6333)

# Charger le modèle DeBERTa
model = SentenceTransformer('microsoft/deberta-v3-large')

# Vérifiez si CUDA est disponible et utilisez le GPU si possible
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = model.to(device)
device = 'cuda' 
model = model.to(device)

# Créer une collection dans Qdrant
client.recreate_collection(
    collection_name="msmarco_segments",
    vectors_config=VectorParams(size=1024, distance='Cosine')
)

# Index BM25
bm25_documents = []
bm25_index = None


# Définition du Dataset personnalisé pour charger les fichiers
class TextDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        data = []
        with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    data.append(doc)
                except json.JSONDecodeError as e:
                    print(f"Erreur de décodage JSON : {e}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Fonction pour indexer les documents locaux
def index_local_documents(directory, batch_size=1000):
    global bm25_index
    bm25_documents = []

    # Listing des fichiers dans le répertoire
    files = os.listdir(directory)

    # Itération à travers chaque fichier
    for file in tqdm(files):
        file_path = os.path.join(directory, file)

        # Chargement des données à partir du fichier
        dataset = TextDataset(file_path)
        total_length = len(dataset)
        num_batches = (total_length + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_length)
            batch = [dataset[idx] for idx in range(start_idx, end_idx) if dataset[idx] is not None]

            if not batch:
                continue

            embeddings = model.encode([doc['segment'] for doc in batch], convert_to_tensor=True).to(device)

            points = []
            for doc, embedding in zip(batch, embeddings):
                embedding = embedding.tolist()
                doc_id = str(uuid.uuid4())

                points.append(PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={"url": doc['url'], "title": doc['title'], "segment": doc['segment']}
                ))

                bm25_documents.append(doc['segment'].split())

            # Upsert des points dans la collection Qdrant
            response = client.upsert(
                collection_name="msmarco_segments",
                points=points
            )
            print(f"Upsert Response: {response}")

    # Création de l'index BM25Okapi une fois tous les documents traités
    bm25_index = BM25Okapi(bm25_documents)
    print("Index BM25 créé avec succès.")


if __name__ == '__main__':
     # Chemin vers le répertoire contenant les fichiers de MS MARCO
    #local_directory = '\\wsl.localhost\Ubuntu-22.04\home\bernie\M1\TER\qdran\msmarco_v2.1_doc_segmented'
    local_directory = r'\\wsl.localhost\Ubuntu-22.04\home\bernie\M1\TER\qdran\msmarco_v2.1_doc_segmented'

    print("Indexation des documents locaux...\n")
    # Indexation des documents locaux
    index_local_documents(local_directory)
    print("Indexation terminée.....\n")
    
    # Exemple de requête de recherche
    query = "Who arrange a deal with Dukat?"
    query_embedding = model.encode(query).tolist()
    query_embedding = model.encode(query, convert_to_tensor=True).to(device).tolist()

    # Recherche initiale avec BM25
    query_tokens = query.split()
    bm25_scores = bm25_index.get_scores(query_tokens)
    top_n_bm25 = bm25_index.get_top_n(query_tokens, bm25_documents, n=10)

    print("Top 10 documents BM25 and scores:")
    for doc, score in top_n_bm25:
        print(f"Document: {doc}, Score: {score}")
        print()

