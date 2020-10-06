from nltk.data import find
import xml.etree.ElementTree as ET
import nltk
import gensim
import sys
import pandas as pd

def get_stop_words():
    '''Get a list with Portuguese and English stop words'''
    stop_words = read_file('./datasets/en_stop_words.txt')
    stop_words += read_file('./datasets/pt_stop_words.txt')
    return stop_words

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().splitlines()
    except IOError as error:
        print("[ERROR] file not found - "+filename)
        sys.exit(1)

def get_kb(filename):
    lines = read_file(filename)
    stopwords = get_stop_words()
    kb = {}
    for l in lines:
        pos = 0
        record = ET.fromstring('<record>'+l+'</record>')
        for segment in record:
            attr = segment.tag
            if attr not in kb:
                kb[attr] = {}
            segment_text = nltk.word_tokenize(segment.text)
            for w in segment_text:
                word = w.lower()
                pos += 1
                if word in stopwords:
                    continue
                if word not in kb[attr]:
                    kb[attr][word] = []
                kb[attr][word].append(pos)
    return kb


def get_probabilities(kb,sentence):
    tokens = nltk.word_tokenize(sentence)
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
    data = {'segments':[],'label':[]}
    for l in lines:
        record = ET.fromstring('<record>'+l+'</record>')
        for segment in record:
            data['segments'].append(segment.text)
            data['label'].append(segment.tag)
    df = pd.DataFrame(data, columns = ['segments', 'label'])
    print(df)


def kb_to_embedding_matrix(kb,filename):
    word2vec_sample = str(find('../../models/word2vec_sample/pruned.word2vec.txt'))
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    features = []
    lines = read_file(filename)
    for l in lines:
        pos_rec = 1
        record = ET.fromstring('<record>'+l+'</record>')
        for segment in record:
            attr = segment.tag
            tokens = nltk.word_tokenize(segment.text)
            tags = nltk.pos_tag(tokens)
            size = len(tokens)
            probs = get_probabilities(kb,segment.text)
            for pos_seg,w in enumerate(tokens):
                word = w.lower()
                vector = []
                if word in model:
                    vector.append(model[word])
                else:
                    vector.append([])
                vector.append(pos_seg)
                vector.append(pos_rec)
                vector.append(size)
                vector.append(tags[pos_seg][1])
                vector.append(probs[word])
                features.append(vector)
                labels.append(attr)
    return features
