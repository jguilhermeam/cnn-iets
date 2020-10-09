from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence, one_hot
from nltk.data import find
import nltk
import gensim
from functions import functions as F
import numpy as np


class CNN(object):

    def __init__(self,kb):
        self.kb = kb
        word2vec_sample = str(find('../../models/word2vec_sample/pruned.word2vec.txt'))
        self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
        #word2vec tem 300 dimensoes

    def create_embedding_matrix(self,df,vocab_size,word_index):
        pos_dict = {}
        embedding_matrix = np.zeros((vocab_size,320))
        for index, row in df.iterrows():
            segment = row['segment']
            attr = row['attribute']
            tokens = text_to_word_sequence(segment)
            pos_tags = nltk.pos_tag(tokens)
            segment_size = len(tokens)
            cp_vector = F.get_probabilities(self.kb,segment)
            for pos_segment,word in enumerate(tokens):
                vector = np.zeros(300)
                #first feature is word2vec
                if word in self.model:
                    vector = self.model[word]
                #second feature is position in segment
                vector = np.append(vector,pos_segment)
                #fourth feature is size of the segment
                vector = np.append(vector,segment_size)
                #fifth feature is pos_tag
                tag = pos_tags[pos_segment][1]
                if tag not in pos_dict:
                    pos_dict[tag] = one_hot(tag, 50)
                vector = np.append(vector,pos_dict[tag])
                #sixth feature is cp propability vector
                vector = np.append(vector,cp_vector[word])
                #third feature is position in record
                pos_records = self.kb[attr][word][:10]
                vector = np.append(vector,pos_records)
                #add to embedding matrix
                vector = F.padarray(vector,320)
                embedding_matrix[word_index[word]] = vector
        return embedding_matrix


    def preprocess(self,df):
        X_train, X_test, y_train, y_test = train_test_split(df['segment'], df['label'], test_size=0.10, random_state=100)        
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(df['segment'])
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        vocab_size = len(tokenizer.word_index) + 1
        maxlen = 8
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
        embedding_matrix = self.create_embedding_matrix(df,vocab_size,tokenizer.word_index)
