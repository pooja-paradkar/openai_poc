import csv
from flask import Flask, request, jsonify, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pptx import Presentation
from pptx.util import Inches
import random 
import pandas as pd
import numpy as np
import os
import re
import tiktoken
from openai import AzureOpenAI
import openai
import logging

app = Flask(__name__)

csv_file_path = 'C:/POC_mastek/bill_sum_data.csv'
api_key = os.getenv('AZURE_OPENAI_API_KEY')
api_version = "2023-05-15"
azure_endpoint = os.getenv('AZURE_ENDPOINT')
logging.basicConfig(filename='error.log', level=logging.ERROR)

print('API_KEY', api_key)

def filter_data(csv_data, filters):
    filtered_data = []

    for row in csv_data:
        if all(row[key] == value for key, value in filters.items()):
            filtered_data.append(row)

    return filtered_data


@app.route('/get_data', methods=['GET'])
def get_data():
    try:
        data = []

        with open('bill_sum_data.csv', 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)

        filters = request.args.to_dict()

        data = filter_data(data, filters)
        return jsonify({'data': data})
    
    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
        return jsonify({'error': str(e)}), 500  # Return an error response with a 500 status code

    
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

client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint)

def generate_embeddings(text, model="Mastek-OpenAI-TE-Deployment-v1"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

df_bills['ada_v2'] = df_bills["text"].apply(lambda x : generate_embeddings (x, model = 'Mastek-OpenAI-TE-Deployment-v1')) 
# model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="Mastek-OpenAI-TE-Deployment-v1"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

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

def search_docs(df, user_query, top_n=4, to_print=True):
    embedding = get_embedding(
        user_query,
        model="Mastek-OpenAI-TE-Deployment-v1" # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )

    return res

client_for_generate_completion = AzureOpenAI(api_key=api_key, api_version="2023-07-01-preview", azure_endpoint="https://mastek-open-ai.openai.azure.com/")

@app.route('/generate_completion', methods=['POST'])
def generate_completion():
    try:
        data = request.get_json()
        user_input = data.get('user_input')

        # Generate completions using Azure OpenAI
        completion = client_for_generate_completion.chat.completions.create(
            model="Mastek-OpenAI-Deployment-v1",
            messages=[
                {
                    "content": user_input,
                    'role':'user'
                },
            ],
        )
        completion_text = completion.choices[0].message.content
        response_data = {'completion': completion_text}
        return jsonify(response_data)
    
    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000,debug=True)