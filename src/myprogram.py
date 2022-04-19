#!/usr/bin/env python
import os
import string
from pickletools import optimize
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
#import pandas as pd
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
#from keras.preprocessing.sequence import pad_sequences
import keras.models
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#import re
#from sklearn.model_selection import train_test_split
import sys
from keras import optimizers
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    def load_training_data(cls,fname):

        text_file = open(fname, 'r',encoding= 'utf-8').read()
        raw_text = text_file.lower()
        raw_text = ''.join(c for c in raw_text if not c.isdigit())

        return raw_text

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp.lower())
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))


    def run_train(self, data, work_dir):

       
        
        chars = sorted(list(set(data)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        int_to_char = dict((i,c) for i, c in enumerate(chars))
        n_chars = len(data) # total characters in entire text
        n_vocab = len(chars) #every possible character
        print('n vocab')
        print(n_vocab)
        
    
        length = 60 # len of input 
        step = 10 
        sentences = [] # x values (sentences)
        next_chars = [] # y values (character that follows sentence)
        for i in range(0, n_chars - length, step):
            sentences.append(data[i: i + length]) # sequence in
            next_chars.append(data[i + length]) #sequence out
        n_patterns = len(sentences)

        x = np.zeros((len(sentences), length,n_vocab), dtype=np.bool)
        y= np.zeros((len(sentences),n_vocab),dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i,t, char_to_int[char]] = 1
            y[i, char_to_int[next_chars[i]]] = 1
        
        
        model = Sequential()
        model.add(LSTM(128, input_shape=(length,n_vocab)))
        model.add(Dense(n_vocab, activation='softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer)
        model.summary()

        checkpoint = ModelCheckpoint(work_dir, monitor='loss',verbose=1,save_best_only=True) #save best only might be prob
        callbacks_list = [checkpoint]

        history = model.fit(x,y,batch_size=128,epochs=5,callbacks=callbacks_list)

        return model


    def run_pred(self, data):

        chars = sorted(list(set(data)))
        #n_vocab = len(chars)
        out_pred = []

        sentence = data[0]
        next_char = ""
        x_pred=np.zeros((1,60,48))
        for t,char in enumerate(sentence):
            x_pred[0,t,mapping[char]] = 1.
        
        preds=model.predict(x_pred,verbose=0)[0]

        preds =np.array(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds/np.sum(exp_preds)
        probas = np.random.multinomial(1,preds,1)

        yhat=np.argmax(probas)
        for char, index in mapping.items():
            if index == yhat:
                next_char = char
                break
        next_char += next_char
        out_pred.append(next_char)
        print('out_pred')
        print(out_pred)

        return out_pred
    



    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        #with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
         #   model.summary(print_fn=lambda x: f.write(x + '\n'))
       
         model.save(work_dir)

        

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        #with open(os.path.join(work_dir, 'model.checkpoint')) as f:
        #    dummy_save = f.read()
        
        loaded_model = load_model(work_dir)
        return loaded_model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work/model.checkpoint')
    parser.add_argument('--train_data', help='path to train data', default='example/train.txt')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        myModel = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data(args.train_data)
        print('Training')
        model= myModel.run_train(train_data, args.work_dir)
        print('Saving model')
        myModel.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        myModel = MyModel()
        mapping = dict((c, i) for i, c in enumerate(MyModel.load_training_data(args.train_data)))
        print('mapping 2:')
        print(mapping)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('test data')
        print(test_data)
        print('Making predictions')
        pred = myModel.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        myModel.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
