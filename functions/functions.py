from nltk.data import find
import xml.etree.ElementTree as ET
import nltk
import gensim
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


def get_probabilities(kb,sentence):
    splited = sentence.split()
    probs = {}

    #get probabilities p(word,attr)
    for i,word in enumerate(splited):
        denominator = 1
        probs[word] = []
        for attr in kb:
            if word in kb[attr]:
                freq = kb[attr][word]
                for x in range(1,freq+1):
                    denominator += 1/x
        for attr in kb:
            numerator = 1
            if word in kb[attr]:
                freq = kb[attr][word]
                if freq > 0:
                    for x in range(1,freq+1):
                        numerator += 1/x
                    probs[word].append(numerator/denominator)
            else:
                probs[word].append(0)
    return probs



def sentence_to_vector(kb,sentence):
    word2vec_sample = str(find('../../models/word2vec_sample/pruned.word2vec.txt'))
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    size = len(sentence.split())
    stopwords = get_stop_words()
    features = []
    probs = get_probabilities(kb,sentence)
    for i,word in enumerate(tokens):
        if word in stopwords:
            continue
        vector = []
        if word in model:
            vector.append(model[word])
        else:
            vector.append([])
        vector.append(i)
        vector.append(size)
        vector.append(tags[i][1])
        vector.append(probs[word])
        features.append(vector)
    return features
