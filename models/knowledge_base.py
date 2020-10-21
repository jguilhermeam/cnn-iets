from utils import functions as F
import xml.etree.ElementTree as ET


class KnowledgeBase:

    def __init__(self, kb_file):
        self.k_base = {}
        self.co_occurrences = {}
        self.inverted_k_base = {}
        self.init_kb(kb_file)
        self.init_inverted_k_base()

    def init_kb(self, kb_file):
        lines = F.read_file(kb_file)
        for l in lines:
            pos = 0
            record = ET.fromstring('<record>'+l+'</record>')
            for segment in record:
                attr = segment.tag
                if attr not in self.k_base:
                    self.k_base[attr] = {}
                terms = F.normalize_str(segment.text).split()
                for term in terms:
                    pos += 1
                    if term not in self.k_base[attr]:
                        self.k_base[attr][term] = []
                    self.k_base[attr][term].append(pos)
                
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
