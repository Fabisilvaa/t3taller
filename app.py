# app.py

from flask import Flask, request, render_template, jsonify
import time
import re
import os
from funciones import relevant, generate_completion, vectorizar, vector_query


app = Flask(__name__)

scripts_dir = 'scripts/'
last_request = 0 
movie_name = None

# Serve HTML page
@app.route('/')
def index():
    return render_template('index.html')


# Entry point for user interaction
@app.route('/ask_chat', methods=['POST'])
def ask_chat():
    global movie_name
    data = request.json
    query= data.get('question', '')

    if re.search(r'\belemental\b', query, re.IGNORECASE):
        movie_name = "ELEMENTAL"
    elif re.search(r'\bcoraline\b', query, re.IGNORECASE):
        movie_name = "CORALINE"
    elif re.search(r'\bthe breakfast club\b', query, re.IGNORECASE):
        movie_name = "THE BREAKFAST CLUB"
    elif re.search(r'\binto the woods\b', query, re.IGNORECASE):
        movie_name = "INTO THE WOODS"
    elif re.search(r'\bthe perks of being a wallflower\b', query, re.IGNORECASE):
        movie_name = "THE PERKS OF BEING A WALLFLOWER"
    elif re.search(r'\bpride and perjudice\b', query, re.IGNORECASE):
        movie_name = "PRIDE AND PERJUDICE"
    elif re.search(r'\bthe addams family\b', query, re.IGNORECASE):
        movie_name = "THE ADDAMS FAMILY"
    elif re.search(r'\bheathers\b', query, re.IGNORECASE):
        movie_name = "HEATHERS"
    elif re.search(r'\bferris bueller\b', query, re.IGNORECASE):
        movie_name = "FERRIS BUELLER'S DAY OFF"
    elif re.search(r'\bfantastic beasts\b', query, re.IGNORECASE):
        movie_name = "FANTASTIC BEASTS"

    response = generate_answer(query, movie_name)  
    return response  #devuelve al frontend la respuesta


@app.route('/ask', methods=['POST'])
def generate_answer(query, movie_name):
    global last_request

    print("segmentando..")
    for filename in os.listdir(scripts_dir):  #1. vectorizar ese script
        if filename.endswith('_segmented.txt') and filename.startswith('cleaned') and (movie_name in filename):
            file_path = os.path.join(scripts_dir, filename)
            movie_seg =  vectorizar(file_path)
        
    query_seg = vector_query(query) 
    print("database armada!")
   
    contexto= relevant(query_seg, movie_seg)

    prompt = f"Context:{contexto}\n\nQuestion:{query}\nAnswer:"

    print(prompt) 
    
    # Rate limiting (10 requests per second)
    current_time = time.time()
    if current_time - last_request < 0.1:  
        time.sleep(0.1 - (current_time - last_request))
    last_request = time.time()

    response= generate_completion(prompt)
    print("la response es: ")
    print(response)
    if 'response' in response and response:
        return jsonify({"answer": response['response']})
    else:
        return jsonify({"error": "Failed to generate a response"}), 500

####################################################

if __name__ == "__main__":

    app.run(debug=True)
