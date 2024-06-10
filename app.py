import logging
import os
from flask import Flask, jsonify
from openai import AzureOpenAI
from flask import Flask

app = Flask(__name__)

csv_file_path = 'C:/POC_mastek/bill_sum_data.csv'
contacts_file_path = 'C:/POC_mastek/Sn_Bot_Contacts.csv'

api_key = os.getenv('AZURE_OPENAI_API_KEY')  #'0aabea4b36aa45a78d1d8a6834ff9f6b'
azure_endpoint_url = os.getenv('AZURE_ENDPOINT') #"https://mastek-open-ai-dev.openai.azure.com/"

api_version_TE = "2023-05-15"
deployment_model_TE = "text-embedding-ada-002-dev"
api_version_GPT ="2023-07-01-preview"
deployment_model_GPT = "gpt-35-turbo-16k-dev"

client = AzureOpenAI(api_key=api_key, api_version=api_version_GPT, azure_endpoint=azure_endpoint_url)
client_for_generate_completion = AzureOpenAI(api_key=api_key, api_version=api_version_GPT, azure_endpoint=azure_endpoint_url)

logging.basicConfig(filename='error.log', level=logging.ERROR)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status='healthy')

# get_contacts.get_contacts()
if __name__ == '__main__':
    app.run(port=5000,debug=True)
    