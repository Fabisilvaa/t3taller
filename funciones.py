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
db_path = "movie_embeddings.db"
client = chromadb.Client(Settings(persist_directory=db_path)) #database para los vectores
collection = client.get_or_create_collection("movie_embeddings")

def vector_query(query):
    embedding = []
    if query:
        query_payload = {
            "model": "nomic-embed-text",
            "input": query
        }

        try:
            query_response = requests.post(
                "http://tormenta.ing.puc.cl/api/embed",
                json=query_payload,
                headers={"Content-Type": "application/json"}
            )

            if query_response.status_code != 200:
                print(f"Non-200 response for query: {query_response.status_code}, Content: {query_response.text}")
            else:
                query_data = query_response.json()
                query_embedding = query_data["embeddings"][0]

                if len(query_embedding) < 768:
                    query_embedding = query_embedding + [0] * (768 - len(query_embedding))
                elif len(query_embedding) > 768:
                    query_embedding = query_embedding[:768]

                embedding.append({
                    "id": str(uuid.uuid4()),
                    "segment": "user_query",
                    "embedding": query_embedding,
                    "metadata": {
                        "filename": "user_query",
                        "segment_index": 0
                    }
                })

        except requests.RequestException as e:
            print(f"Request error embedding user query: {e}")
    return embedding[0]

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

############################################ CONSULTAR
def relevant(query, collection_name="movie_embeddings", max_chars=500):

    contexto = []

    try:
        results = collection.query(
            query_embeddings=[query["embedding"]],
            n_results=5,  #numero de segmentos max
        )
        print("buscando texto original")
        for original_text in results['documents']:

            if len(original_text) > 768:  #maximo largo del contexto
                original_text = original_text[:768]
            print(original_text)
                
            contexto.append(original_text)
            
    except Exception as e:
        print(f"Error querying collection: {e}")
        return []

    return contexto


#hacer embed al prompt, busca los embed similares, pasa eso de vuelta a string y eso lo madno de contexto, poner la pelicyla en la metadat del embed al armas la abse de dtos
##############################################   RESPUESTA DEL MODELO
#def generate_completion(prompt):#

 # url = "http://tormenta.ing.puc.cl/api/generate"

  #data = {
   # "model": "integra-LLM",
   # "prompt": prompt,
   #3 "temperature": 6,
   # "top_k": 18,
  #}

  #headers = {
  #  "Content-Type": "application/json"
  #}

 # response = requests.post(url, headers=headers, json=data)

  #if response.status_code == 200:
  #  response_data = response.json()
  #  return response_data['choices'][0]['text'].strip()

  #else:

    #print(f"Error al llamar a la API: {response.text}")
    #return None




def generate_completion(prompt):  #generate response
    url = "http://tormenta.ing.puc.cl/api/generate"
    payload = {
        "model": "integra-LLM",
        "prompt": prompt,
        "temperature": 6,
        "num_ctx": 2048,
        "repeat_last_n": 10,
        "top_k": 18,
        "stream": False 

    }
    headers = {
        "Content-Type": "application/json"
    }

    try:

        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("Request timed out. Try reducing the prompt size or retrying.")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    return None

def generate_chat_completion(messages):  #chat response
    url = "http://tormenta.ing.puc.cl/api/chat" 
    payload = {
        "model": "integra-LLM",
        "messages": messages,  
        "temperature": 6,
        "num_ctx": 2048,
        "repeat_last_n": 10,
        "top_k": 18,
        "stream": False  
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()  
        return response.json() 
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None 