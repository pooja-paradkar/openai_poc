import logging
import csv
from flask import Flask, request, jsonify
from openai import AzureOpenAI
from app import client, api_key, azure_endpoint_url, api_version_GPT, deployment_model_GPT

app = Flask(__name__)

logging.basicConfig(filename='error.log', level=logging.ERROR)

def filter_data(csv_data, filters):
    filtered_data = []

    for row in csv_data:
        if all(row[key] == value for key, value in filters.items()):
            filtered_data.append(row)

    return filtered_data

# Function to get similar words
def get_synonyms(word):
    # Generate synonyms for the given word
    response = client.chat.completions.create(
        model=deployment_model_GPT,  # Adjust the model according to your needs
        messages=[{"content": f"Generate synonyms for: {word}", "role": "user"}],
        max_tokens=30,
        n=5,
        stop=None,
        temperature=0.5
    )
    synonyms = [choice.message.content.strip() for choice in response.choices]
    return synonyms

def expand_search_terms(filters):
    expanded_filters = {}
    for key, value in filters.items():
        synonyms = get_synonyms(value)
        synonyms.append(value)  # Add original search term as well
        expanded_filters[key] = synonyms
    return expanded_filters

@app.route('/get_contacts', methods=['GET'])
def get_contacts():
    try:
        data = []
        with open('Sn_Bot_Contacts.csv', 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)

        filters = request.args.to_dict()

        expanded_filters = expand_search_terms(filters)

        data = filter_data(data, expanded_filters)
        return jsonify({'data': data})
    
    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(port=5000,debug=True)