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

def label_anchor_blocks(k_base,records,threshold):
    print("Labelling anchor blocks...")
    kb = k_base.k_base
    for blocks in records:
        anchors = {}
        for attr in kb:
            anchors[attr] = []
        for i,block in enumerate(blocks):
            word = block.value
            denominator = 1
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
                        prob = (numerator/denominator)
                        if prob > threshold:
                            if len(anchors[attr]) > 0:
                                if prob > anchors[attr][0].anchor_prob:
                                    if freq > anchors[attr][0].freq:
                                        anchors[attr][0].clear()
                                        block.set_anchor(attr,prob,freq)
                                        anchors[attr][0] = block
                                    elif freq == anchors[attr][0].freq:
                                        block.set_anchor(attr,prob,freq)
                                        anchors[attr].append(block)
                            else:
                                block.set_anchor(attr,prob,freq)
                                anchors[attr].append(block)
        #now we fix anchors for each record
        fix_anchor_blocks(blocks,anchors)

def fix_anchor_blocks(blocks,anchors):
    for attr in anchors:
        if len(anchors[attr]) > 1:
            print("CASO ESPECIAL CORRIGIR")

def get_max_prob(partition,probs):
    max = -1
    choosen = None
    for k in probs:
        if k[0] > max and k[1] in partition:
            max = k[0]
            choosen = k
    return choosen

def greedy_labelling(blocks,probs,threshold):
    partitions = [[blocks[0]]]
    j = 0
    for i in range(1,len(blocks)):
        if blocks[i].is_anchor() and blocks[i-1].is_anchor():
            partitions.append([blocks[i]])
            j += 1
        else:
            partitions[j].append(blocks[i])
    for p in partitions:
        while True:
            choosen = get_max_prob(p,probs)
            if choosen[0] <= threshold:
                break
            probs.remove(choosen)
            sc = choosen[1]
            Ac = choosen[2]
            sc.label = Ac
            index = p.index(sc)
            for i,sc_b in enumerate(p):
                if i < index and sc_b.label == Ac:
                    in_between = False
                    for j in range(i+1,index):
                        if p[j].label != Ac and len(p[j].label) > 0:
                            in_between = True
                    if in_between == True:
                        sc.label = ''
                    else:
                        for j in range(i+1,index):
                            p[j].label = Ac
                elif i > index and sc_b.label == Ac:
                    in_between = False
                    for j in range(index+1,i):
                        if p[j].label != Ac and len(p[j].label) > 0:
                            in_between = True
                    if in_between == True:
                        sc.label = ''
                    else:
                        for j in range(index+1,i):
                            p[j].label = Ac
        if len(probs) == 0:
            break
        exists_unlabelled = False
        for b in p:
            if len(b.label) == 0:
                exists_unlabelled = True
        if exists_unlabelled == False:
            break




def get_dataset(kb_file,num_classes):
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
    code_labels = le.inverse_transform(range(0,num_classes))
    return df,code_labels
