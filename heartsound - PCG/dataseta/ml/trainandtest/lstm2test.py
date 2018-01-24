
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
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
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

with open('a/traindata.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

trainX = np.array(your_list)

traindata = pd.read_csv('a/trainlabels.csv', header=None)
Y = traindata.iloc[:,0]
y_train1 = np.array(Y)
y_train= to_categorical(y_train1)

maxlen = 44100
trainX = sequence.pad_sequences(trainX, maxlen=maxlen)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))



with open('a/testdata.csv', 'rb') as f:
    reader1 = csv.reader(f)
    your_list1 = list(reader1)

testX = np.array(your_list1)

testdata = pd.read_csv('a/testlabels.csv', header=None)
Y1 = testdata.iloc[:,0]
y_test1 = np.array(Y1)
y_test= to_categorical(y_test1)

maxlen = 44100
testX = sequence.pad_sequences(testX, maxlen=maxlen)

# reshape input to be [samples, time steps, features]
X_test = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



batch_size = 2

model = Sequential()
model.add(LSTM(32,input_dim=44100)) 
model.add(Dropout(0.1))
#model.add(LSTM(512, return_sequences=True))
#model.add(Dropout(0.1))
#model.add(LSTM(512, return_sequences=False))
#model.add(Dropout(0.1))
#model.add(LSTM(512, return_sequences=False))
#model.add(Dropout(0.1))
model.add(Dense(4))
model.add(Activation('softmax'))
 
import os
for file in os.listdir("logs/lstm1layer/"):
  model.load_weights("logs/lstm1layer/"+file)
  y_pred = model.predict_classes(X_test)
  print(file)
  print("---------------------------------------------------------------------------------")
  accuracy = accuracy_score(y_test1, y_pred)
  recall = recall_score(y_test1, y_pred , average="weighted")
  precision = precision_score(y_test1, y_pred , average="weighted")
  f1 = f1_score(y_test1, y_pred, average="weighted")

  print("accuracy")
  print("%.3f" %accuracy)
  print("precision")
  print("%.3f" %precision)
  print("recall")
  print("%.3f" %recall)
  print("f1score")
  print("%.3f" %f1)
