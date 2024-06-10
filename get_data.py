import logging
import csv
from flask import Flask, request, jsonify

app = Flask(__name__)

logging.basicConfig(filename='error.log', level=logging.ERROR)

def filter_data(csv_data, filters):
    filtered_data = []

    for row in csv_data:
        if all(row[key] == value for key, value in filters.items()):
            filtered_data.append(row)

    return filtered_data

#Fetching data from local
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(port=5000,debug=True)