import xml.etree.ElementTree as ET
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.preprocessing.text import text_to_word_sequence

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().splitlines()
    except IOError as error:
        print("[ERROR] file not found - "+filename)
        sys.exit(1)

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

def get_kb(filename):
    lines = read_file(filename)
    kb = {}
    for l in lines:
        pos = 0
        record = ET.fromstring('<record>'+l+'</record>')
        for segment in record:
            attr = segment.tag
            if attr not in kb:
                kb[attr] = {}
            segment_text = text_to_word_sequence(segment.text)
            for w in segment_text:
                word = w.lower()
                pos += 1
                if word not in kb[attr]:
                    kb[attr][word] = []
                kb[attr][word].append(pos)
    return kb


def get_probabilities(kb,sentence):
    tokens = text_to_word_sequence(sentence)
    probs = {}

    #get probabilities p(word,attr)
    for i,w in enumerate(tokens):
        word = w.lower()
        denominator = 1
        probs[word] = []
        for attr in kb:
            if word in kb[attr]:
                freq = len(kb[attr][word])
                for x in range(1,freq+1):
                    denominator += 1/x
        for attr in kb:
            numerator = 1
            if word in kb[attr]:
                freq = len(kb[attr][word])
                if freq > 0:
                    for x in range(1,freq+1):
                        numerator += 1/x
                    probs[word].append(numerator/denominator)
            else:
                probs[word].append(0)
    return probs

def get_dataset(filename):
    lines = read_file(filename)
    data = {'segment':[],'attribute':[]}
    for l in lines:
        record = ET.fromstring('<record>'+l+'</record>')
        for segment in record:
            data['segment'].append(segment.text)
            data['attribute'].append(segment.tag)
    df = pd.DataFrame(data, columns = ['segment','attribute','label'])
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['attribute'])
    return df

