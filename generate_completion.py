import logging
from flask import Flask, request, jsonify
from app import client, deployment_model_GPT



app = Flask(__name__)

logging.basicConfig(filename='error.log', level=logging.ERROR)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status='healthy')


@app.route('/generate_completion', methods=['POST'])
def generate_completion():
    try:
        data = request.get_json()
        user_input = data.get('user_input')

        # Generate completions using Azure OpenAI
        completion = client.chat.completions.create(
            model=deployment_model_GPT,
            messages=[
                {
                    "content": user_input,
                    'role': 'user'
                },
            ],
        )
        completion_text = completion.choices[0].message.content
        response_data = {'Completion': completion_text}
        return jsonify(response_data)
    
    
    except Exception as e:
        logging.error(f'An error occurred while generating completion: {str(e)}')
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(port=5000,debug=True)