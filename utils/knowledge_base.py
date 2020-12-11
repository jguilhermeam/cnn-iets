from utils import functions as F
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn import preprocessing
import numpy as np

class KnowledgeBase:

    def __init__(self, kb_file):
        self.k_base = {}
        self.registers = set()
        self.df = None
        self.labels_dict = None
        print("Loading KB...")
        self.init_kb(kb_file)
        self.create_dataframe(kb_file)
        self.num_attributes = len(self.get_attributes())

    def get_attributes(self):
        return list(self.k_base.keys())

    def init_kb(self, kb_file):
        try:
            tree = ET.parse(kb_file)
        except ET.ParseError as error:
            print("Error reading KB file. Cause: "+error.msg)
            sys.exit(1)
        data = tree.getroot()
        for segment in data:
            attr = segment.tag
            text = F.normalize_str(segment.text)
            self.registers.add(text)
            if attr not in self.k_base:
                self.k_base[attr] = {}
            terms = text.split()
            for term in terms:
                if term not in self.k_base[attr]:
                    self.k_base[attr][term] = 0
                self.k_base[attr][term] += 1


    def create_dataframe(self,kb_file):
        num_attributes = len(self.get_attributes())
        try:
            tree = ET.parse(kb_file)
        except ET.ParseError as error:
            print("Error reading KB file for Pandas Dataframe. Cause: "+error.msg)
            sys.exit(1)
        record = tree.getroot()
        data = {'segment':[],'attribute':[]}
        for segment in record:
            data['segment'].append(F.normalize_str(segment.text))
            data['attribute'].append(segment.tag)
        self.df = pd.DataFrame(data, columns = ['segment','attribute','label'])
        le = preprocessing.LabelEncoder()
        self.df['label'] = le.fit_transform(self.df['attribute'])
        self.labels_dict = le.inverse_transform(range(0,num_attributes))

    def get_probabilities(self,terms):
        probabilities = {}
        attributes = self.get_attributes()
        #get probabilities p(word,attr)
        for i,word in enumerate(terms):
            denominator = 1
            probabilities[word] = np.zeros(self.num_attributes)
            for attr in self.k_base:
                if word in self.k_base[attr]:
                    freq = self.k_base[attr][word]
                    for x in range(1,freq+1):
                        denominator += 1/x
            for i,attr in enumerate(attributes):
                numerator = 0
                if word in self.k_base[attr]:
                    freq = self.k_base[attr][word]
                    for x in range(1,freq+1):
                        numerator += 1/x
                    if numerator > 0:
                        numerator += 1 #to avoid false 1.0 probabilities
                    probabilities[word][i] = numerator/denominator
        return probabilities
