import re
import chromadb
from chromadb import Client
import json
import os
import uuid 
import requests
import time
from bs4 import BeautifulSoup

scripts_dir = 'scripts'
MODEL_NAME = "integra-LLM"

def getcollection(collection_name="movie_script_embeddings"):
    client = Client()
    collection = client.get_collection(collection_name)
    return collection 


########################################## LIMPIAR DATOS
def clean_script(file_path): 

    with open(file_path, 'r', encoding='utf-8') as file: #leer archivo
        text = file.read()
    
    soup = BeautifulSoup(text, 'html.parser') #sacar HTML tags
    text = soup.get_text()

    text = re.sub(r'[^A-Za-z0-9\s.,?!\'"\-]', '', text) #usa regex para sacar caracteres especiales

    text = re.sub(r'\s+', ' ', text)  #saca espacios libres extras
    text = re.sub(r'\n+', '\n', text) 
    text = text.strip() 

    cleaned_lines = []
    for line in text.split('\n'):

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
    return segments

   

############################### 
###############################
###############################
if __name__ == "__main__":
    
    # Perform initial setup if running this file directly
    for filename in os.listdir(scripts_dir):
        if filename.endswith('.txt') and not filename.startswith('cleaned_'):
            file_path = os.path.join(scripts_dir, filename)

            # Clean the script and save it
            cleaned_text = clean_script(file_path)
            cleaned_filename = os.path.join(scripts_dir, f'cleaned_{filename}')
            with open(cleaned_filename, 'w', encoding='utf-8') as cleaned_file:
                cleaned_file.write(cleaned_text)

            # Segment text and embed it in the collection
            segment_text(cleaned_filename)

