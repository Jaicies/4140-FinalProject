#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
import re
from sklearn.model_selection import train_test_split

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    def load_training_data(cls,fname):

      #  data = []
       # with open(fname) as f:
        #    for line in f:
         #       inp = line[:-1]  # the last character is a newline
          #      data.append(inp)
        text_file = open(fname)
        data = text_file.read()
        text_file.close()

        newString = data.lower()
        newString = re.sub(r"'s\b","",newString)
        # remove punctuations
        newString = re.sub("[^a-zA-Z]", " ", newString) 
        long_words=[]
        # remove short words
        for i in newString.split():
            if len(i)>=3:                  
                long_words.append(i)
        data_new = (" ".join(long_words)).strip()
        #print("new data: " + data_new)
        return data_new


    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):

        length = 30
        sequences = list()
        for i in range(length, len(data)):
            # select sequence of tokens
            seq = data[i-length:i+1]
            # store
            sequences.append(seq)
        print('Total Sequences: %d' % len(sequences))
        
        chars = sorted(list(set(data)))
        mapping = dict((c, i) for i, c in enumerate(chars))

        new_seq = list()
        for line in sequences:
            # integer encode line
            encoded_seq = [mapping[char] for char in line]
            # store
            new_seq.append(encoded_seq)

        vocab = len(mapping)
        new_seq = np.array(new_seq)
        # create X and y
        X, y = new_seq[:,:-1], new_seq[:,-1]
        # one hot encode y
        y = to_categorical(y, num_classes=vocab)
        # create train and validation sets
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        #print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)
                
        model = Sequential()
        model.add(Embedding(vocab, 50, input_length=30, trainable=True))
        model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
        model.add(Dense(vocab, activation='softmax'))
        #print(model.summary())

        # compile the model
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
        # fit the model
        model.fit(X_tr, y_tr, epochs=3, verbose=2, validation_data=(X_val, y_val))

        return mapping,model
        

    def run_pred(self, data):

        seq_length = len(data)
        in_text = data
	# generate a fixed number of characters
        for _ in range(3):
            
            # encode the characters as integers
            encoded = [mapping[char] for char in in_text]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
            # predict character
            yhat = myModel.model.predict_classes(encoded, verbose=0)
            # reverse map integer to character
            out_char = ''
            for char, index in mapping.items():
                if index == yhat:
                    out_char = char
                    break
            # append to output
            preds += char
            # probably : in_text += out_char
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
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
        holder = myModel.run_train(train_data, args.work_dir)
        mapping = holder[0]
        model = holder[1]
        print('Saving model')
        myModel.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        myModel = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = myModel.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        myModel.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
