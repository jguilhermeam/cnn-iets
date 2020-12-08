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
from utils.knowledge_base import KnowledgeBase as KB
import numpy as np


class CNN(object):

    def __init__(self):
        word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        self.wv = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
        self.wv_size = 300

    def create_weights(self,word_list,word_dict,attributes,k_base):
        vocab_size = len(word_dict)+1
        self.wv_matrix = np.zeros((vocab_size,self.wv_size))
        self.cp_size = len(attributes)
        self.cp_matrix = np.zeros((vocab_size,self.cp_size))
        for words in word_list:
            for w in words:
                if w in self.wv:
                    vector = self.wv[w]
                else:
                    vector = np.random.rand(self.wv_size)
                self.wv_matrix[word_dict[w]] = vector
                self.cp_matrix[word_dict[w]] = k_base.get_probabilities(w)

    def define_and_train(self,df,k_base,attributes):
        word_list = []
        positions_list = []
        lengths_list = []
        pos_tags_list = []
        for index, row in df.iterrows():
            words = row['segment'].split()
            word_list.append(words)
            positions_list.append([i+1 for i in range(len(words))])
            lengths_list.append([len(words)])
            tags = [i[1] for i in nltk.pos_tag(words)]
            pos_tags_list.append(tags)


        self.word_dict = F.makeDictFromList(word_list)
        self.pos_tags_dict = F.makeDictFromList(pos_tags_list)

        words_mapped = F.mapWordToId(word_list,self.word_dict)
        pos_tags_mapped = F.mapWordToId(pos_tags_list,self.pos_tags_dict)

        words_mapped, seq_len = F.pad(words_mapped)
        pos_tags_mapped, _ = F.pad(pos_tags_mapped)
        positions_mapped, _ = F.pad(positions_list)
        lengths_mapped = F.pad_to(lengths_list,seq_len)

        self.create_weights(word_list,self.word_dict,attributes,k_base)
        self.max_len = seq_len

        vocab_size = len(self.word_dict)+1
        pos_tag_vocab_size = len(self.pos_tags_dict)+1

        input1 = Input(shape=(seq_len,))
        input2 = Input(shape=(seq_len,))
        input3 = Input(shape=(seq_len,))
        input4 = Input(shape=(seq_len,))
        input5 = Input(shape=(seq_len,))
        emb1 = Embedding(vocab_size, self.wv_size, weights=[self.wv_matrix], input_length=seq_len, trainable=True)(input1)
        emb2 = Embedding(seq_len+1, 5, input_length=seq_len, trainable=True)(input2)
        emb3 = Embedding(seq_len+1, 5, input_length=seq_len, trainable=True)(input3)
        emb4 = Embedding(pos_tag_vocab_size, 5, input_length=seq_len, trainable=True)(input4)
        emb5 = Embedding(vocab_size, self.cp_size, weights=[self.cp_matrix], input_length=seq_len, trainable=True)(input5)
        embedding = concatenate([emb1,emb2,emb3,emb4,emb5])
        layers = []
        for i in [4,6]:
            conv = Conv1D(filters=128, kernel_size=i, activation='relu')(embedding)
            poolsize = seq_len-i+1
            pool = MaxPooling1D(pool_size=poolsize)(conv)
            layers.append(pool)
        # merge
        merged = concatenate(layers)
        #flatten and Dropout
        flat = Flatten()(merged)
        drop = Dropout(0.5)(flat)
        # softmax
        outputs = Dense(len(attributes), activation='softmax')(drop)
        self.model = Model(inputs=[input1,input2,input3,input4,input5], outputs=outputs)
        # compile
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # summarize
        print(self.model.summary())
        #train
        self.model.fit(x=[words_mapped,positions_mapped,lengths_mapped,pos_tags_mapped,words_mapped], y=df['label'], epochs=10, verbose=0)

        self.labels_dict = k_base.labels_dict

    def predict(self,words):
        terms = words.split()
        words_mapped = F.mapWordToId([terms],self.word_dict)
        words_mapped = F.pad_to(words_mapped,self.max_len)
        positions = [[i+1 for i in range(len(terms))]]
        positions = F.pad_to(positions,self.max_len)
        length = [[len(terms)]]
        length = F.pad_to(length,self.max_len)
        tags = [i[1] for i in nltk.pos_tag(terms)]
        pos_tags_mapped = F.mapWordToId([tags],self.pos_tags_dict)
        pos_tags_mapped = F.pad_to(pos_tags_mapped,self.max_len)
        predictions = self.model.predict(x=[words_mapped,positions,length,pos_tags_mapped,words_mapped])[0]
        scores = {}
        for i in range(0,len(predictions)):
            scores[self.labels_dict[i]] = predictions[i]
        return scores
