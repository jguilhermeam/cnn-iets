import sys
import pandas as pd
import numpy as np
import unicodedata
import re
import xml.etree.ElementTree as ET
from sklearn import preprocessing
from keras.preprocessing.text import text_to_word_sequence

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
    if re.match(r'^\d+\s\d+$', normalized):
        return re.sub(r' ', '', normalized)
    return normalized

def get_probabilities(k_base,terms,num_classes):
    probs = {}
    kb = k_base.k_base
    #get probabilities p(word,attr)
    for i,word in enumerate(terms):
        denominator = 1
        probs[word] = np.zeros(num_classes)
        for attr in kb:
            if word in kb[attr]:
                freq = kb[attr][word]
                for x in range(1,freq+1):
                    denominator += 1/x
        for i,attr in enumerate(kb):
            numerator = 1
            if word in kb[attr]:
                freq = kb[attr][word]
                if freq > 0:
                    for x in range(1,freq+1):
                        numerator += 1/x
                    probs[word][i] = numerator/denominator
    return probs

def get_dataset(kb_file):
    try:
        tree = ET.parse(kb_file)
    except ET.ParseError as error:
        print("Error reading KB file for Pandas Dataframe. Cause: "+error.msg)
        sys.exit(1)
    record = tree.getroot()
    data = {'segment':[],'attribute':[]}
    for segment in record:
        data['segment'].append(normalize_str(segment.text))
        data['attribute'].append(segment.tag)
    df = pd.DataFrame(data, columns = ['segment','attribute','label'])
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['attribute'])
    return df
