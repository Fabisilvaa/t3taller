import re
import chromadb
from chromadb import Client
from chromadb.config import Settings
import uuid 
import os
import requests
from bs4 import BeautifulSoup

scripts_dir = 'scripts'
MODEL_NAME = "integra-LLM"
db_path = 'db' 
client = chromadb.Client(Settings(persist_directory=db_path,is_persistent=True )) #database para los vectores
collection = client.get_or_create_collection("movie_embeddings")


def vectorizar(file_path): #para un solo script
    with open(file_path, 'r', encoding='utf-8') as file: #leer archivo
        text = file.read()

    ###### separa en segmentos por escenas
    separator = re.split(r'(?=\bEXT\.|\bINT\.)', text)  #separador por escena
    segments = [scene.strip() for scene in separator if scene.strip()]

    embeddings = []

    for i, segment in enumerate(segments):
        if len(segment) > 768:  #maximo largo del contexto
            segment = segment[:768]

        payload = {
            "model": "nomic-embed-text",
            "input": segment
        }
        print(segment)
        
        try:
            response = requests.post(
                "http://tormenta.ing.puc.cl/api/embed",
                json= payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code != 200:
                print(f"Non-200 response: {response.status_code}, Content: {response.text}")
                continue

            response_data = response.json()
            embedding = response_data["embeddings"][0]

            embeddings.append({
                "id": str(uuid.uuid4()), 
                "segment": segment,
                "embedding": embedding,
                "metadata": {
                    "filename": file_path, #the movie name
                    "segment_index": i
                }
            })

        except requests.RequestException as e:
            print(f"Request error embedding segment {i} of file {file_path}: {e}")
            continue

    ###### insertar vectores en la database
    if embeddings:
        try:
            collection.add(
                ids=[data["id"] for data in embeddings],
                documents=[data["segment"] for data in embeddings],
                embeddings=[data["embedding"] for data in embeddings],
                metadatas=[data["metadata"] for data in embeddings]
            )
        except Exception as e:
            print(f"Error saving to vector database: {e}")
    return embeddings


movie_name="HEATHERS"
for filename in os.listdir(scripts_dir):  #1. vectorizar ese script
            if filename.endswith('_segmented.txt') and filename.startswith('cleaned') and (movie_name in filename):
                file_path = os.path.join(scripts_dir, filename)
                movie_seg =  vectorizar(file_path)

print("cree la database!")