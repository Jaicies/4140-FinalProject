#!/usr/bin/env python
import os
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


class MyModel:
    @classmethod
    def load_training_data(cls,fname):
        # read file and preprocess
        text_file = open(fname, 'r',encoding= 'utf-8').read()
        raw_text = text_file.lower()
        raw_text = ''.join(c for c in raw_text if not c.isdigit())

        return raw_text

    @classmethod
    def load_test_data(cls, fname):
        # IF test data is entered via text file - not used in current implementation
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp.lower())
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        #write predictions to file - not necessary in current implementation
        with open(fname, 'wt') as f:
          f.write('{}\n'.format(preds))


    def run_train(self, data, work_dir):
        
        chars = sorted(list(set(data)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        print('mapping 1')
        print(char_to_int)
        n_chars = len(data) # total characters in entire text
        n_vocab = len(chars) #every possible character
        print('n vocab')
        print(n_vocab)
        
    
        length = 30 # len of input 
        step = 5 # jump ever 5 characters
        sentences = [] # x values (sentences)
        next_chars = [] # y values (character that follows sentence)
        for i in range(0, n_chars - length, step):
            sentences.append(data[i: i + length]) # sequence in
            next_chars.append(data[i + length]) #sequence out
       

        x = np.zeros((len(sentences), length,n_vocab), dtype=np.bool)
        y= np.zeros((len(sentences),n_vocab),dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i,t, char_to_int[char]] = 1
            y[i, char_to_int[next_chars[i]]] = 1
        
        #create model
        model = Sequential()
        model.add(LSTM(128, input_shape=(length,n_vocab)))
        model.add(Dense(n_vocab, activation='softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer)
        model.summary()

        checkpoint = ModelCheckpoint(work_dir, monitor='loss',verbose=1,save_best_only=True) #save best only might be prob
        callbacks_list = [checkpoint]

        history = model.fit(x,y,batch_size=128,epochs=30,callbacks=callbacks_list)

        return model


    def run_pred(self, data=None):

        if data is None:
            return('first iteration!') # for now

        sentence =""
        sentence = data # just doing first sentence for now
        
        predictions = ""
        x_pred=np.zeros((1,30,n_vocab))
        for t,char in enumerate(sentence):
            x_pred[0,t,mapping[char]] = 1.
        
        preds=model.predict(x_pred,verbose=0)[0]  # return probability values for each of the 35 characters

        top_values_index = sorted(range(len(preds)), key=lambda i: preds[i])[-3:]

        for yhat in top_values_index:
            for char, index in mapping.items():
                if index == yhat:
                    predictions += char
        
        print(predictions)

        return predictions


    def save(self, work_dir):
         model.save(work_dir)

        

    @classmethod
    def load(cls, work_dir):
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
        test_data = ""
        userInput = ""
        myModel = MyModel()
        while True: 
            print('Loading model')
            model = MyModel.load(args.work_dir)
            data = MyModel.load_training_data(args.train_data)
            mapping = dict((c, i) for i, c in enumerate(sorted(list(set(data)))))
            n_vocab = len(sorted(list(set(data))))
            test_data = test_data + userInput.lower()
            print('test data:')
            print(test_data)
            print('Making predictions')
            pred = myModel.run_pred(test_data)
            print('Writing predictions to {}'.format(args.test_output))
            userInput = input('please select a character or hit ctrl + c to exit ')
            #assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
            myModel.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
