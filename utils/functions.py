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
    #if re.match(r'^\d+\s\d+$', normalized):
    #    normalized = re.sub(r' ', '', normalized)
    normalized = re.sub('\d','dg',normalized) #change digits do dg
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

def label_anchor_blocks(k_base,blocks,threshold):
    kb = k_base.k_base
    anchors = {}
    for attr in kb:
        anchors[attr] = []
    for i,block in enumerate(blocks):
        denominator = 1
        for attr in kb:
            for word in block.value.split():
                if word in kb[attr]:
                    freq = kb[attr][word]
                    for x in range(1,freq+1):
                        denominator += 1/x
        for attr in kb:
            numerator = 0
            for word in block.value.split():
                if word in kb[attr]:
                    freq = kb[attr][word]
                    for x in range(1,freq+1):
                        numerator += 1/x
            if numerator > 0:
                numerator += 1
            prob = (numerator/denominator)
            if prob > threshold:
                if len(anchors[attr]) > 0:
                    if prob > anchors[attr][0].anchor_prob:
                        anchors[attr][0].clear()
                        block.set_anchor(attr,prob,1)
                        anchors[attr] = [block]
                else:
                    block.set_anchor(attr,prob,1)
                    anchors[attr].append(block)
    #TODO fix anchors

def get_missing_anchors(record,k_base):
    missing = list(k_base.keys())
    for block in record:
        if block.is_anchor == True:
            missing.remove(block.label)
    return missing

def adjust_cnn_probs(probs,blocks,i,missing):
    possible_attributes = []
    for j in reversed(range(0,i)):
        if blocks[j].is_anchor == True:
            possible_attributes.append(blocks[j].label)
            break
    for j in range(i+1,len(blocks)):
        if blocks[j].is_anchor == True:
            possible_attributes.append(blocks[j].label)
            break
    possible_attributes.extend(missing)
    new_probs = []
    denominator = 0
    for attr in possible_attributes:
        denominator += probs[attr]
    for attr in possible_attributes:
        p = probs[attr]/denominator
        new_probs.append((p,blocks[i],attr))
    return new_probs



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
        if blocks[i].is_anchor == True and blocks[i-1].is_anchor == True:
            partitions.append([blocks[i]])
            j += 1
        else:
            partitions[j].append(blocks[i])
    for p in partitions:
        while True:
            choosen = get_max_prob(p,probs)
            if choosen == None or choosen[0] <= threshold:
                break
            probs.remove(choosen)
            sc = choosen[1]
            Ac = choosen[2]
            sc.label = Ac
            index = blocks.index(sc)
            for i,sc_b in enumerate(blocks):
                if i < index and sc_b.label == Ac:
                    in_between = False
                    for j in range(i+1,index):
                        if blocks[j].label != Ac and len(blocks[j].label) > 0:
                            in_between = True
                    if in_between == True:
                        sc.label = 'none'
                    else:
                        for j in range(i+1,index):
                            blocks[j].label = Ac
                elif i > index and sc_b.label == Ac:
                    in_between = False
                    for j in range(index+1,i):
                        if blocks[j].label != Ac and len(blocks[j].label) > 0:
                            in_between = True
                    if in_between == True:
                        sc.label = 'none'
                    else:
                        for j in range(index+1,i):
                            blocks[j].label = Ac
            if len(probs) == 0:
                break
            exists_unlabelled = False
            for b in p:
                if b.label == 'none':
                    exists_unlabelled = True
                    break
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
