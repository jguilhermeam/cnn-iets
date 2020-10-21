from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
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
from nltk.data import find
import nltk
import gensim
from utils import functions as F
from models.knowledge_base import KnowledgeBase as KB
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
            terms = F.normalize_str(segment)
            pos_tags = nltk.pos_tag(terms)
            segment_size = len(terms)
            cp_vector = F.get_probabilities(self.kb,terms)
            for pos_segment,word in enumerate(terms):
                vector = np.zeros(300)
                #first feature is word2vec
                if word in self.model:
                    vector = self.model[word]
                #second feature is position in segment
                vector = np.append(vector,pos_segment)
                #third feature is position in record
                pos_records = self.kb.k_base[attr][word][:1]
                vector = np.append(vector,pos_records)
                #fourth feature is size of the segment
                vector = np.append(vector,segment_size)
                #fifth feature is pos_tag
                tag = pos_tags[pos_segment][1]
                if tag not in pos_dict:
                    pos_dict[tag] = one_hot(tag, 50)
                vector = np.append(vector,pos_dict[tag])
                #sixth feature is cp propability vector
                vector = np.append(vector,cp_vector[word])
                #add to embedding matrix
                vector = F.padarray(vector,320)
                embedding_matrix[word_index[word]] = vector
        return embedding_matrix

    def define_model(self, num_classes, length, vocab_size,embedding_matrix):
        inputs = Input(shape=(length,))
        embedding = Embedding(vocab_size, 320, weights=[embedding_matrix], input_length=8, trainable=True)(inputs)
        # channel 1
        conv1 = Conv1D(filters=320, kernel_size=4, activation='relu')(embedding)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=5)(drop1)
        flat1 = Flatten()(pool1)
        # channel 2
        conv2 = Conv1D(filters=320, kernel_size=6, activation='relu')(embedding)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=3)(drop2)
        flat2 = Flatten()(pool2)
        # merge
        merged = concatenate([flat1, flat2])
        # interpretation
        outputs = Dense(num_classes, activation='softmax')(merged)
        model = Model(inputs=inputs, outputs=outputs)
        # compile
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # summarize
        print(model.summary())
        return model


    def preprocess(self,df):
        X_train, X_test, y_train, y_test = train_test_split(df['segment'], df['label'], test_size=0.30, random_state=100)        
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(df['segment'])
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        vocab_size = len(tokenizer.word_index) + 1
        maxlen = 8
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
        embedding_matrix = self.create_embedding_matrix(df,vocab_size,tokenizer.word_index)
        # define model
        model = self.define_model(4,maxlen,vocab_size,embedding_matrix)
        # fit the model
        model.fit(X_train, y_train, epochs=10, verbose=0)
        # evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print('Accuracy: %f' % (accuracy*100))
