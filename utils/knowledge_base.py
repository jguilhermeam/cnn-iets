from utils import functions as F
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn import preprocessing
import numpy as np

class KnowledgeBase:

    def __init__(self, kb_file):
        self.k_base = {}
        self.co_occurrences = {}
        self.inverted_k_base = {}
        self.df = None
        self.labels_dict = None
        print("Loading KB...")
        self.init_kb(kb_file)
        self.init_inverted_k_base()
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
            if attr not in self.k_base:
                self.k_base[attr] = {}
            terms = F.normalize_str(segment.text).split()
            for term in terms:
                if term not in self.k_base[attr]:
                    self.k_base[attr][term] = 0
                self.k_base[attr][term] += 1

            i = 0
            while i < len(terms)-1:
                if terms[i] in self.co_occurrences:
                    if (terms[i+1], attr) not in self.co_occurrences[terms[i]]:
                        self.co_occurrences[terms[i]].append((terms[i+1], attr))
                else:
                    self.co_occurrences[terms[i]] = []
                i += 1

            if terms[-1] not in self.co_occurrences:
                self.co_occurrences[terms[-1]] = []



    def init_inverted_k_base(self):
        for attribute in self.k_base:
            for term in self.k_base[attribute]:
                if term not in self.inverted_k_base:
                    self.inverted_k_base[term] = {}
                if attribute not in self.inverted_k_base[term]:
                    self.inverted_k_base[term][attribute] = self.k_base[attribute][term]

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

    def get_probabilities(self,word):
        probabilities = np.zeros(self.num_attributes)
        attributes = self.get_attributes()
        denominator = 1
        for attr in attributes:
            if word in self.k_base[attr]:
                freq = self.k_base[attr][word]
                for x in range(1,freq+1):
                    denominator += 1/x
        for i,attr in enumerate(self.labels_dict):
            numerator = 0
            if word in self.k_base[attr]:
                freq = self.k_base[attr][word]
                for x in range(1,freq+1):
                    numerator += 1/x
                if numerator > 0:
                    numerator += 1 #to avoid false 1.0 probabilities
                probabilities[i] = numerator/denominator
        return probabilities
