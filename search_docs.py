import os
import re
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import tiktoken
from app import client, csv_file_path, deployment_model_TE, deployment_model_GPT



app = Flask(__name__)

logging.basicConfig(filename='error.log', level=logging.ERROR)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status='healthy')

df = pd.read_csv(os.path.join(os.getcwd(),csv_file_path))
df_bills = df[['text','summary', 'title']]

pd.options.mode.chained_assignment = None

def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "") 
    s = s.strip()
    
    return s

df_bills['text']= df_bills["text"].apply(lambda x : normalize_text(x))

tokenizer = tiktoken.get_encoding("cl100k_base")
df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
df_bills = df_bills[df_bills.n_tokens<8192]

sample_encode = tokenizer.encode(df_bills.text[0]) 
decode = tokenizer.decode_tokens_bytes(sample_encode)

def generate_embeddings(text, model=" "): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

df_bills['ada_v2'] = df_bills["text"].apply(lambda x : generate_embeddings (x, model = deployment_model_TE)) 
# model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model=deployment_model_GPT): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_docs(df, user_query, top_n=4, to_print=True):
    embedding = get_embedding(
        user_query,
        model=deployment_model_TE # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )

    return res

@app.route('/search_docs', methods=['GET'])
def search_docs_route():
    try:
        # Get user_query from request body
        data = request.get_json()
        user_query = data.get('user_query')

        # Call search_docs function with user_query
        res = search_docs(df_bills, user_query)
        
        # Extract relevant information for response
        response_data = [{'text': row['text'], 'summary': row['summary'], 'title': row['title']} for _, row in res.iterrows()]

        return jsonify({'search_results': response_data})

    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(port=5000,debug=True)