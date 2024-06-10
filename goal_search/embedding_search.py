import os
import logging
import pandas as pd
import numpy as np
import json
from flask import Flask, request, jsonify
from app import client, deployment_model_TE, deployment_model_GPT

app = Flask(__name__)

# Load the processed CSV file
csv_file_path = 'goals_embedding_processed.csv'
df_goals = pd.read_csv(os.path.join(os.getcwd(), csv_file_path))

# Parse the JSON strings back into lists of floats
df_goals['name_embedding'] = df_goals['name_embedding'].apply(json.loads)
df_goals['roles_embedding'] = df_goals['roles_embedding'].apply(json.loads)

logging.basicConfig(filename='error.log', level=logging.DEBUG)  # Set to DEBUG level

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model=deployment_model_GPT):  # model = "deployment_name"
    return client.embeddings.create(input=[text], model=model).data[0].embedding

@app.route('/search_goals', methods=['POST'])
def search_goals_route():
    try:
        # Get user_query and role from request body
        data = request.get_json()
        user_query = data.get('user_query')
        role = data.get('role')

        if not user_query or not role:
            return jsonify({'error': 'Both user_query and role are required.'}), 400

        # Call search_goals function with user_query and role
        res = search_goals(df_goals, user_query, role)

        # Extract relevant information for response
        seen = set()
        response_data = []

        for _, row in res.iterrows():
            record = {'name': row['name'], 'metric': row['metric'], 'weight': row['weight'], 'roles': row['roles']}
            record_tuple = tuple(record.items())  # Convert the dictionary to a tuple of items to make it hashable
            if record_tuple not in seen:
                seen.add(record_tuple)
                response_data.append(record)

        return jsonify({'search_results': response_data})

    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
        return jsonify({'error': str(e)}), 500


def search_goals(df, user_query, role):
    logging.debug(f"User query: {user_query}, Role: {role}")
    
    # Generate embedding for the user-provided name query
    name_embedding = get_embedding(user_query, model=deployment_model_TE)

    # Generate                     for the user-provided role
    role_embedding = get_embedding(role, model=deployment_model_TE)

    # Filter DataFrame based on role
    df["role_similarities"] = df['roles_embedding'].apply(lambda x: cosine_similarity(x, role_embedding))
    df_filtered = df[df["role_similarities"] > 0.5]  # Adjust this threshold as needed
    logging.debug(f"Number of rows after role filtering: {df_filtered.shape[0]}")

    # Calculate similarities for the name query within the filtered DataFrame
    df_filtered["similarities"] = df_filtered['name_embedding'].apply(lambda x: cosine_similarity(x, name_embedding))

    # Sort the results by similarity and drop duplicate names
    res1 = (
        df_filtered.sort_values("role_similarities", ascending=False)
        ).head(50)
    
    res2 = res1.sort_values("similarities", ascending=False).head(10)
#     # Return the top_n results
#     res = df_filtered

    return res2

# def search_goals(df, user_query, role):
 
#    # Generate embedding for the user-provided name query
#     name_embedding = get_embedding( user_query, model=deployment_model_TE)
   
#    # Generate embedding for the user-provided role
#     role_embedding = get_embedding(role, model=deployment_model_TE)
#     df_filtered = df[df["role_similarities"] > 0.5]
 
#     # Filter DataFrame based on role and names
#     df["role_similarities"] = df.roles_embedding.apply(lambda x: cosine_similarity(x, role_embedding))    
#     df["similarities"] = df.name_embedding.apply(lambda x: cosine_similarity(x, name_embedding))
   
#     res1 = (
#         df.sort_values("role_similarities", ascending=False)
#         .head(50)
#     )
 
#     res2 = (
#         res1.sort_values("similarities", ascending=False)
#         .head(10)
#     )
 
#     return res2

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(port=5000, debug=True)
