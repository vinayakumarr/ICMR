from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D

traindata = pd.read_csv('dataset/traindata.csv', header=None)
testdata = pd.read_csv('dataset/testdata.csv', header=None)


X = traindata.iloc[:,1:15002]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:15002]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))

batch_size = 4

# 1. define the network
cnn = Sequential()
cnn.add(Convolution1D(32, 3, border_mode="same",activation="relu",input_shape=(15000, 1)))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu"))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(Flatten())
cnn.add(Dense(32, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(1, activation="sigmoid"))

# try using different optimizers and different optimizer configs
cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/cnn1layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
cnn.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, callbacks=[checkpointer])
cnn.save("logs/cnnlayer/cnnlayer_model.hdf5")

