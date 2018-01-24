from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from keras import backend as K
from sklearn.cross_validation import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
import numpy as np
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import csv

with open('trainnormal.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

trainX = np.array(your_list)
#print(X)

traindata = pd.read_csv('trainlabels.csv', header=None)
Y = traindata.iloc[:,0]
y_train1 = np.array(Y)
y_train= to_categorical(y_train1)

maxlen = 100000
trainX = sequence.pad_sequences(trainX, maxlen=maxlen)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))



with open('testnormal.csv', 'rb') as f:
    reader1 = csv.reader(f)
    your_list1 = list(reader1)

testX = np.array(your_list1)
#print(X)

testdata = pd.read_csv('testlabels.csv', header=None)
Y1 = traindata.iloc[:,0]
y_test1 = np.array(Y1)
y_test= to_categorical(y_test1)

testX = sequence.pad_sequences(testX, maxlen=maxlen)

# reshape input to be [samples, time steps, features]
X_test = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))




# 1. define the network
model = Sequential()
model.add(LSTM(64,input_dim=100000))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(LSTM(64,return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(LSTM(64,return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(LSTM(64, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(LSTM(64, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(LSTM(64, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(LSTM(64, return_sequences=False))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(3))
model.add(Activation('softmax'))

# define optimizer and objective, compile cnn

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

# train
checkpointer = callbacks.ModelCheckpoint(filepath="results/lstm1results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('results/lstm1results/cnntrainanalysis1.csv',separator=',', append=False)
model.fit(X_train, y_train, nb_epoch=1000, show_accuracy=True,validation_split= 0.20,callbacks=[checkpointer,csv_logger])


