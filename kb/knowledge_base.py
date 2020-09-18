import xml.etree.ElementTree as ET
import sys

def get_stop_words():
    '''Get a list with Portuguese and English stop words'''
    try:
        with open('./datasets/en_stop_words.txt', 'r') as f:
            stop_words = f.read().splitlines()
        with open('./datasets/pt_stop_words.txt', 'r') as f:
            stop_words += f.read().splitlines()
        return stop_words
    except IOError as error:
        print("stopwords files were not found")
        sys.exit(1)

def get_kb(filename):
    try:
        kb_pointer = ET.parse(filename).getroot()
    except FileNotFoundError:
        print(filename+" does not exist ")
        exit()
    kb = {}
    stopwords = get_stop_words()
    for item in kb_pointer:
        attribute = item.tag
        segments = item.text.split()
        if attribute not in kb:
            kb[attribute] = {}
        for s in segments:
            word = s.lower()
            if word in stopwords:
                continue
            if word not in kb[attribute]:
                kb[attribute][word] = 0
            kb[attribute][word] += 1
    return kb
