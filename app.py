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
import pika
import threading
import uuid
from time import sleep
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
# Disable SSL verification (only use this in development)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__, static_url_path='', static_folder='.')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# SQLite Database setup
DATABASE = 'chatbot.db'
PDF_FOLDER = 'pdfFolder'

# Initialize Redis client
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
try:
    redis_client.ping()
    print("Redis is connected!")
except redis.ConnectionError:
    print("Could not connect to Redis.")

# RabbitMQ setup
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', '127.0.0.1')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'guest')
RABBITMQ_PASS = os.getenv('RABBITMQ_PASS', 'guest')
QUERY_QUEUE = 'query_queue'
RESPONSE_QUEUE = 'response_queue'

# Global variables for text chunks and vectorizer
text_chunks = []
vectorizer = None
chunk_embeddings = None

def delete_queues():
    """Delete existing queues if they exist"""
    try:
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            connection_attempts=3,
            retry_delay=1
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        
        # Delete existing queues
        channel.queue_delete(queue=QUERY_QUEUE)
        channel.queue_delete(queue=RESPONSE_QUEUE)
        
        connection.close()
        logging.info("Successfully deleted existing queues")
    except Exception as e:
        logging.warning(f"Error deleting queues (this is normal if queues don't exist): {e}")

def setup_rabbitmq():
    """Setup RabbitMQ connection with retry logic"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Create connection parameters explicitly using IPv4
            parameters = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                connection_attempts=3,
                retry_delay=1,
                socket_timeout=5
            )
            
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            # Declare queues (non-durable for development)
            channel.queue_declare(queue=QUERY_QUEUE)
            channel.queue_declare(queue=RESPONSE_QUEUE)
            
            logging.info("Successfully connected to RabbitMQ")
            return connection, channel
            
        except pika.exceptions.AMQPConnectionError as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to connect to RabbitMQ after {max_retries} attempts: {e}")
                raise
            logging.warning(f"RabbitMQ connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
            sleep(retry_delay)

def get_rabbitmq_channel():
    """Get a new channel with error handling"""
    try:
        connection, channel = setup_rabbitmq()
        return connection, channel
    except Exception as e:
        logging.error(f"Failed to get RabbitMQ channel: {e}")
        raise

def publish_response(channel, correlation_id, response):
    """Publish response to RabbitMQ response queue with error handling"""
    try:
        channel.basic_publish(
            exchange='',
            routing_key=RESPONSE_QUEUE,
            properties=pika.BasicProperties(
                correlation_id=correlation_id
            ),
            body=json.dumps({'response': response})
        )
        logging.info(f"Published response for correlation_id: {correlation_id}")
    except Exception as e:
        logging.error(f"Error publishing response: {e}")
        raise

def process_query(ch, method, properties, body):
    """Process incoming queries from RabbitMQ"""
    try:
        query_data = json.loads(body)
        question = query_data['question']
        correlation_id = properties.correlation_id
        
        logging.info(f"Processing query with correlation_id: {correlation_id}")
        
        # Check Redis cache first
        response = get_cached_response(question)
        if response is not None:
            logging.info("Response retrieved from Redis cache.")
        else:
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
        
        # Publish response back to RabbitMQ
        publish_response(ch, correlation_id, response)
        
        # Acknowledge the message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        # Negative acknowledge the message to requeue it
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def start_consuming():
    """Start consuming messages from RabbitMQ in a separate thread with reconnection logic"""
    while True:
        try:
            connection, channel = setup_rabbitmq()
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=QUERY_QUEUE, on_message_callback=process_query)
            
            logging.info("Starting to consume messages from RabbitMQ")
            channel.start_consuming()
            
        except pika.exceptions.ConnectionClosedByBroker:
            logging.warning("Connection was closed by broker, retrying...")
            continue
            
        except pika.exceptions.AMQPChannelError as e:
            logging.error(f"Channel error: {e}, stopping...")
            break
            
        except pika.exceptions.AMQPConnectionError:
            logging.warning("Connection was lost, retrying...")
            continue
            
        except Exception as e:
            logging.error(f"Unexpected error in consumer thread: {e}")
            if 'connection' in locals() and connection and not connection.is_closed:
                connection.close()
            sleep(5)
            continue

# Delete existing queues before starting
delete_queues()

# Start consumer thread
consumer_thread = threading.Thread(target=start_consuming)
consumer_thread.daemon = True
consumer_thread.start()

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
    text = re.sub('\s+', ' ', text)
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
    try:
        question = request.form.get('question')
        correlation_id = str(uuid.uuid4())
        
        # Setup RabbitMQ connection for publishing
        connection, channel = get_rabbitmq_channel()
        
        # Publish question to query queue
        channel.basic_publish(
            exchange='',
            routing_key=QUERY_QUEUE,
            properties=pika.BasicProperties(
                correlation_id=correlation_id,
                reply_to=RESPONSE_QUEUE
            ),
            body=json.dumps({'question': question})
        )
        
        # Setup consumer for this specific response
        response = None
        def callback(ch, method, properties, body):
            if properties.correlation_id == correlation_id:
                nonlocal response
                response = json.loads(body)
                channel.stop_consuming()
        
        # Start consuming from response queue
        channel.basic_consume(
            queue=RESPONSE_QUEUE,
            on_message_callback=callback,
            auto_ack=True
        )
        
        # Wait for response with timeout
        try:
            channel.start_consuming()
        except Exception as e:
            logging.error(f"Error while consuming response: {e}")
            response = {'error': 'Failed to get response, please try again'}
        
        # Close connection
        connection.close()
        
        return jsonify(response or {'error': 'No response received'})
        
    except Exception as e:
        logging.error(f"Error in ask endpoint: {e}")
        return jsonify({'error': 'An error occurred, please try again'})

if __name__ == '__main__':
    init_db()
    initialize_vectorizer()
    app.run(debug=True)
