from flask import Flask, request, jsonify, render_template
import sqlite3
import redis
import json
import os
import ssl
from openai import OpenAI
import re
import pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Disable SSL verification (only use this in development)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__, static_url_path='', static_folder='.')

# Initialize OpenAI client
client = OpenAI(api_key=<palce your api key here>)

# SQLite Database setup
DATABASE = 'chatbot.db'
PDF_FOLDER = 'pdfFolder'

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
try:
    redis_client.ping()
    print("Redis is connected!")
except redis.ConnectionError:
    print("Could not connect to Redis.")

# Global variables for text chunks and vectorizer
text_chunks = []
vectorizer = None
chunk_embeddings = None
def clear_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('DELETE FROM chat_history')
    conn.commit()
    conn.close()
    print("SQLite database cleared for testing.")

clear_db()
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY, query TEXT, response TEXT)')
    conn.commit()
    conn.close()

def get_response_from_db(query):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT response FROM chat_history WHERE query = ?', (query,))
    response = c.fetchone()
    conn.close()
    return response[0] if response else None

def store_query_response(query, response):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (query, response) VALUES (?, ?)', (query, response))
    conn.commit()
    conn.close()

def get_cached_response(query):
    cached_response = redis_client.get(query)
    if cached_response:
        return json.loads(cached_response)
    return None

def cache_response(query, response, ttl=3600):
    redis_client.setex(query, ttl, json.dumps(response))

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text

def pdf_to_text(path):
    try:
        doc = pymupdf.open(path)
        text_list = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")
            text = preprocess(text)
            if text.strip():  # Only add non-empty pages
                text_list.append(f"[Page {page_num + 1}] {text}")
        
        doc.close()
        return text_list
    except Exception as e:
        print(f"Error processing PDF {path}: {str(e)}")
        return []

def text_to_chunks(text_list, chunk_size=1000):
    chunks = []
    for page_text in text_list:
        page_num = page_text[:page_text.find(']') + 1]
        text = page_text[page_text.find(']') + 1:]
        
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(f"{page_num} {chunk}")
    
    return chunks

def initialize_vectorizer():
    global vectorizer, chunk_embeddings, text_chunks
    
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        return "No PDFs found in the folder. Please add PDFs to the 'pdfFolder' directory."
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        return "No PDFs found in the folder. Please add PDFs to the 'pdfFolder' directory."
    
    all_texts = []
    for file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, file)
        texts = pdf_to_text(str(pdf_path))
        all_texts.extend(texts)
    
    if not all_texts:
        return "Could not extract text from PDFs. Please check if the files are valid."
    
    text_chunks = text_to_chunks(all_texts)
    
    vectorizer = TfidfVectorizer()
    chunk_embeddings = vectorizer.fit_transform(text_chunks)
    
    return "Vectorizer initialized successfully"

def find_relevant_chunks(query, top_k=5):
    if vectorizer is None or chunk_embeddings is None:
        initialize_vectorizer()
    
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_embeddings)
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    
    return [text_chunks[i] for i in top_indices]

def generate_text(prompt):
    try:
        full_response = ""
        while True:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document excerpts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=750,
                temperature=0.8
            )
            
            part = response.choices[0].message.content.strip()
            full_response += part

            if response.choices[0].finish_reason == "stop":
                break
            
            prompt = " ".join(part.split()[-10:]) + " Please continue."

        return full_response
    except Exception as e:
        return str(e)

def ask_file(question: str) -> str:
    if vectorizer is None:
        result = initialize_vectorizer()
        if result != "Vectorizer initialized successfully":
            return result
    
    relevant_chunks = find_relevant_chunks(question, top_k=3)
    
    if not relevant_chunks:
        return "No relevant information found in the PDFs."
    
    prompt = "Based on the following excerpts from documents, please answer the question. Include page numbers when citing information.\n\n"
    for chunk in relevant_chunks:
        prompt += f"{chunk}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    
    response = generate_text(prompt)
    return response

@app.route('/')
def index():
    init_db()
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    
    # Check Redis cache first
    response = get_cached_response(question)
    if response is not None:
        logging.info("Response retrieved from Redis cache.")
        return jsonify({'response': response})

    # Check SQLite database
    response = get_response_from_db(question)
    if response:
        logging.info("Response retrieved from SQLite database.")
    else:
        # If not in Redis or SQLite, call OpenAI API
        response = ask_file(question)
        if response and response != "No relevant information found in the PDFs.":
            store_query_response(question, response)
            cache_response(question, response)
            logging.info("Response generated by OpenAI API and stored in Redis cache.")

    return jsonify({'response': response})
if __name__ == '__main__':
    init_db()
    initialize_vectorizer()
    app.run(debug=True)
