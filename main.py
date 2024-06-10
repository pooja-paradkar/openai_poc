import logging
import app
import get_contacts
from flask import Flask

main = Flask(__name__)
logging.basicConfig(filename='error.log', level=logging.ERROR)

def main():
    app.health_check()
    get_contacts.get_contacts()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()