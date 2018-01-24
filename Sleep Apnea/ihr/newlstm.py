from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.cross_validation import train_test_split

traindata = pd.read_csv('data/newtrain.csv', header=None)

train, test = train_test_split(traindata, train_size = 0.5)

X = train.iloc[:,1:61]
Y = train.iloc[:,0]
C = test.iloc[:,0]
T = test.iloc[:,1:61]



scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])

scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


y_train = np.array(Y)
y_test = np.array(C)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


batch_size = 16

# 1. define the network
model = Sequential()
model.add(LSTM(32,input_dim=60))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/lstm1/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc', mode='max')
csv_logger = CSVLogger('logs/lstm1/training_set_iranalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5000, validation_split=0.33,callbacks=[checkpointer,csv_logger])
model.save("logs/lstm1/lstm1layer_model.hdf5")
'''

import os
for file in os.listdir("logs/lstm1/"):
  model.load_weights("logs/lstm1/"+file)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  loss, accuracy = model.evaluate(X_test, y_test)
  print(file)
  print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

'''











