# app.py

from flask import Flask, request, render_template, jsonify
import os
import json
import time
from fixscripts import segment_text,  clean_script, relevant, generate_completion, generate_chat_completion


app = Flask(__name__)

scripts_dir = 'scripts/'
last_request = 0 

# Serve HTML page
@app.route('/')
def index():
    return render_template('index.html')


# Process scripts and store embeddings in the database
def process_scripts():
    for filename in os.listdir(scripts_dir):
        if filename.endswith('.txt') and not filename.startswith('cleaned_'):
            file_path = os.path.join(scripts_dir, filename)
            
            cleaned_text = clean_script(file_path)
        
            cleaned_filename = os.path.join(scripts_dir, f'cleaned_{filename}')
            with open(cleaned_filename, 'w', encoding='utf-8') as cleaned_file:
                cleaned_file.write(cleaned_text)
                        
            segment_text(cleaned_filename)


# Entry point for user interaction
@app.route('/ask_chat', methods=['POST'])
def ask_chat():
    data = request.json
    user_question = data.get('question', '')

    # Get relevant context and pass to LLM
    response = generate_answer(user_question)  # Reuses the ask_question function to keep flow consistent

    return response  # Returns answer or error as JSON for the frontend


@app.route('/ask', methods=['POST'])
def generate_answer(user_question):
    global last_request

    # 1. Retrieve relevant context from RAG
    context = relevant(user_question)

    # 2. Combine user question with retrieved context for LLM
    prompt = f"Context: {context}\n\nUser Question: {user_question}\nAnswer:"
    
    # Rate limiting (10 requests per second)
    current_time = time.time()
    if current_time - last_request < 0.1:  # 0.1 sec = 10 req/sec
        time.sleep(0.1 - (current_time - last_request))
    last_request = time.time()

    response= generate_chat_completion(prompt)
    if response and 'text' in response:
        return jsonify({"answer": response['text']})
    else:
        return jsonify({"error": "Failed to generate a response"}), 500

####################################################

if __name__ == "__main__":

    process_scripts()
    app.run(debug=True)
