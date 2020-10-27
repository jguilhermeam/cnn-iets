from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import Input
from keras.layers import MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import load_model
from nltk.data import find
import nltk
import gensim
import os.path
from utils import functions as F
from models.knowledge_base import KnowledgeBase as KB
import numpy as np


class CNN(object):

    def __init__(self,k_base,df,num_classes):
        word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        self.wv = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
        self.wv_size = 300 #word2vec tem 300 dimensoes
        self.emb_size = 0
        self.pos_dict = {}
        self.pos_i = 1
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(df['segment'])
        if os.path.isfile("model.h5"):
            self.model = load_model("model.h5")
            self.max_length = 10
        else:
            self.train_model(k_base,df,num_classes)



    def create_embedding_matrix(self,k_base,df,vocab_size,word_index,num_classes):
        pos_dict = {}
        self.emb_size = self.wv_size + num_classes + 3 #word2vec + cp vector + 3 features (position,lenght and postag)
        embedding_matrix = np.zeros((vocab_size,self.emb_size))
        for index, row in df.iterrows():
            segment = row['segment']
            attr = row['attribute']
            terms = segment.split()
            pos_tags = nltk.pos_tag(terms)
            segment_size = len(terms)
            cp_vector = F.get_probabilities(k_base,terms,num_classes)
            for pos_segment,word in enumerate(terms):
                vector = np.zeros(self.wv_size)
                #first feature is word2vec
                if word in self.wv:
                    vector = self.wv[word]
                #second feature is position in segment
                vector = np.append(vector,pos_segment)
                #third feature is position in record - it was removed

                #fourth feature is size of the segment
                vector = np.append(vector,segment_size)
                #fifth feature is pos_tag
                tag = pos_tags[pos_segment][1]
                if tag not in self.pos_dict:
                    self.pos_dict[tag] = self.pos_i
                    self.pos_i += 1
                vector = np.append(vector,self.pos_dict[tag])
                #sixth feature is cp propability vector
                vector = np.append(vector,cp_vector[word])
                #add to embedding matrix
                embedding_matrix[word_index[word]] = vector
        return embedding_matrix


    def define_model(self, num_classes, vocab_size, num_filters, filter_sizes, embedding_matrix):
        inputs = Input(shape=(max_length,))
        embedding = Embedding(vocab_size, self.emb_size, weights=[embedding_matrix], input_length=self.max_length, trainable=True)(inputs)
        layers = []
        for i in filter_sizes:
            conv = Conv1D(filters=num_filters, kernel_size=i, activation='relu')(embedding)
            poolsize = self.max_length-i+1
            pool = MaxPooling1D(pool_size=poolsize)(conv)
            layers.append(pool)
        # merge
        merged = concatenate(layers)
        #flatten and Dropout
        flat = Flatten()(merged)
        drop = Dropout(0.5)(flat)
        # softmax
        outputs = Dense(num_classes, activation='softmax')(drop)
        model = Model(inputs=inputs, outputs=outputs)
        # compile
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # summarize
        print(model.summary())
        return model


    def train_model(self,k_base,df,num_classes):
        X_train, X_test, y_train, y_test = train_test_split(df['segment'], df['label'], test_size=0.10, random_state=100)
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_test = self.tokenizer.texts_to_sequences(X_test)
        vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = -1
        i = 1
        for s in df['segment']:
            self.max_length = max(max_length,len(s.split()))
        X_train = pad_sequences(X_train, padding='post', maxlen=self.max_length)
        X_test = pad_sequences(X_test, padding='post', maxlen=self.max_length)
        embedding_matrix = self.create_embedding_matrix(k_base,df,vocab_size,self.tokenizer.word_index,num_classes)
        # define model
        self.model = self.define_model(num_classes,vocab_size,128,[4,6],embedding_matrix)
        # fit the model
        self.model.fit(X_train, y_train, epochs=10, verbose=0)
        self.model.save("model.h5")
        # evaluate the model
        #loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        #print('Accuracy: %f' % (accuracy*100))

    def predict(self,segment):
        tokens = self.tokenizer.texts_to_sequences(segment.split())
        return self.model.predict(pad_sequences(tokens,padding='post',maxlen=self.max_length))
