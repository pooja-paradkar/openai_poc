import os
import re
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import tiktoken
import json
from app import client, deployment_model_TE, deployment_model_GPT

app = Flask(__name__)

csv_file_path = 'Mastek_goals_data.csv'
processed_csv_file_path = 'goals_embedding_processed.csv'

logging.basicConfig(filename='error.log', level=logging.DEBUG)  # Set to DEBUG level

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status='healthy')

# Load and preprocess the CSV data
df_goals = pd.read_csv(csv_file_path, usecols=['name', 'metric', 'weight', 'roles'])

def normalize_text(s):
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace("..", ".").replace(". .", ".").replace("\n", "").strip()
    return s

df_goals['name'] = df_goals["name"].apply(normalize_text)
df_goals['roles'] = df_goals['roles'].apply(normalize_text)

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Encode the text and filter out rows with too many tokens
df_goals['n_tokens'] = df_goals["name"].apply(lambda x: len(tokenizer.encode(x)))
df_goals = df_goals[df_goals.n_tokens < 8192]

# Generate embeddings for columns
def generate_embeddings(text, model=deployment_model_TE):
    return client.embeddings.create(input=[text], model=model).data[0].embedding

df_goals['name_embedding'] = df_goals["name"].apply(lambda x: generate_embeddings(x))
df_goals['roles_embedding'] = df_goals["roles"].apply(lambda x: generate_embeddings(x))

# Convert embeddings to JSON strings and save to CSV
df_goals['name_embedding'] = df_goals['name_embedding'].apply(json.dumps)
df_goals['roles_embedding'] = df_goals['roles_embedding'].apply(json.dumps)
df_goals.to_csv(processed_csv_file_path, index=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(port=5000, debug=True)
