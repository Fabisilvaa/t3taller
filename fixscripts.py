import re
import chromadb
from chromadb import Client
import json
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import uuid 
import requests
import time
from bs4 import BeautifulSoup

scripts_dir = 'scripts'
MODEL_NAME = "integra-LLM"
client = chromadb.Client() #database para los vectores
collection = client.get_or_create_collection("movie_script_embeddings")

########################################## LIMPIAR DATOS
def clean_script(file_path): 

    with open(file_path, 'r', encoding='utf-8') as file: #leer archivo
        text = file.read()
    
    soup = BeautifulSoup(text, 'html.parser') #sacar HTML tags
    text = soup.get_text()

    text = re.sub(r'[^A-Za-z0-9\s.,?!\'"\-]', '', text) #usa regex para sacar caracteres especiales

   # text = re.sub(r'(\b\d+\b\.?)|(\(\d+\))', '', text) #sacar los numeros de escena

    text = re.sub(r'\s+', ' ', text)  #saca espacios libres extras
    text = re.sub(r'\n+', '\n', text) 
    text = text.strip() 

    cleaned_lines = []
    for line in text.split('\n'):
        #line = re.sub(r'\[.*?\]', '', line).strip() #saca las direcciones de brackets

        line = re.sub(r'[^A-Za-z0-9\s.,?!\'"\-]', '', line).strip()

        if line.isupper(): #estandarizar nombres de personajes
            cleaned_lines.append(f"\n{line}\n") 
        else:
            cleaned_lines.append(line)
    cleaned_text = '\n'.join(cleaned_lines)
    
    return cleaned_text

################################################# SEGMENTAR DATOS Y VECTORIZAR

def segment_text(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as file: #leer archivo
        text = file.read()


    ###### separa en segmentos por escenas
    separator = re.split(r'(?=\bEXT\.|\bINT\.)', text)  #separador por escena
    segments = [scene.strip() for scene in separator if scene.strip()]

    segmented_file_path = file_path.replace('.txt', '_segmented.txt')
    with open(segmented_file_path, 'w', encoding='utf-8') as out_file:
        for i, segment in enumerate(segments):
            out_file.write(f"Scene {i + 1}:\n{segment}\n{'='*40}\n")  #divisor por escena
    print("llegue1")
    
    ###### vectorizar
    embeddings = []
    #session = requests.Session()
    #retry = Retry(
    #    total=3,  # Retry up to 3 times
    #    backoff_factor=1,  # Wait 1 second between retries
    #    status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP codes
    #)
    #adapter = HTTPAdapter(max_retries=retry)
    #session.mount("http://", adapter)

    print("me conecte")

    for i, segment in enumerate(segments):
        payload = {
            "model": "nomic-embed-text",
            "input": segment
        }
        
        try:
            response = requests.post(
                "http://tormenta.ing.puc.cl/api/embed",
                json= payload,
                headers={"Content-Type": "application/json"}
            )

            print("obtuve respuesta")

            if response.status_code != 200:
                print(f"Non-200 response: {response.status_code}, Content: {response.text}")
                continue

            response_data = response.json()
            embedding = response_data["embeddings"][0]

            if len(embedding) < 768:
                embedding = embedding + [0] * (768 - len(embedding))
            elif len(embedding) > 768:
                embedding = embedding[:768]

            embeddings.append({
                "id": str(uuid.uuid4()), 
                "segment": segment,
                "embedding": embedding,
                "metadata": {
                    "filename": file_path,
                    "segment_index": i
                }
            })

        except requests.RequestException as e:
            print(f"Request error embedding segment {i} of file {file_path}: {e}")
            continue
    print("termine de segmentar")

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
    print("arme la abse de datos")

    return segments
   

############################################ CONSULTAR

def relevant(query, collection_name="movie_script_embeddings", max_chars=500):
    client = Client()
    collection = client.get_collection(collection_name)

    payload = {
        "model": "nomic-embed-text",
        "input": query
    }
    
    try:
        response = requests.post(
            "http://tormenta.ing.puc.cl/api/embed",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"Error embedding query: {response.status_code}, Content: {response.text}")
            return ""
        
        response_data = response.json()
        query_embedding = response_data["embeddings"][0]

        # Ensure embedding is exactly 768 dimensions
        if len(query_embedding) < 768:
            query_embedding = query_embedding + [0] * (768 - len(query_embedding))
        elif len(query_embedding) > 768:
            query_embedding = query_embedding[:768]

    except requests.RequestException as e:
        print(f"Request error embedding query: {e}")
        return ""

    # Step 2: Query the collection with the embedded query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )


    #Concatenate relevant results into context, ensuring it stays within max_chars
    context = ""
    for result in results['documents']:
        if len(context) + len(result) <= max_chars:
            context += f"- {result}\n"  # Format each result for better readability
        else:
            break

    # Return context as a trimmed, relevant string for inclusion in the LLM prompt
    return context.strip()


##############################################  
def generate_completion(prompt):
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

def generate_chat_completion(messages):
    url = "http://tormenta.ing.puc.cl/api/chat"  # Using the chat completion endpoint
    payload = {
        "model": "integra-LLM",
        "messages": messages,  # Messages must be a list of dictionaries with 'role' and 'content'
        "temperature": 6,
        "num_ctx": 2048,
        "repeat_last_n": 10,
        "top_k": 18,
        "stream": False  # Disable streaming for a single response
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None  # Return None or handle the error as needed