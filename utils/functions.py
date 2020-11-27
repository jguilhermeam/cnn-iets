import sys
import unicodedata
import re
import xml.etree.ElementTree as ET

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().splitlines()
    except IOError as error:
        print("[ERROR] file not found - "+filename)
        sys.exit(1)

def normalize_str(input_str):
    '''Transform string to lowercase, remove special chars,
    accents and trailing spaces'''
    byte_string = unicodedata.normalize('NFD', input_str.lower()).encode('ASCII', 'ignore')
    normalized = re.sub(r'[^A-Za-z0-9\s]+', '',byte_string.decode('utf-8')).strip()
    #if re.match(r'^\d+\s\d+$', normalized):
    #    normalized = re.sub(r' ', '', normalized)
    normalized = re.sub('\d','dg',normalized) #change digits do dg
    return normalized
