from get_contacts import get_synonyms, expand_search_terms, get_contacts
from get_data import get_data


def main():
    get_synonyms()
    get_contacts()
    get_data()
    expand_search_terms()

if __name__ == "__main__":
    main()