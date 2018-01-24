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


testdata = pd.read_csv('dataset/testdata.csv', header=None)


C = testdata.iloc[:,0]
T = testdata.iloc[:,1:15002]

scaler = Normalizer().fit(T)
testT = scaler.transform(T)


y_test = np.array(C)


# reshape input to be [samples, time steps, features]
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))

batch_size = 4

# 1. define the network
cnn = Sequential()
cnn.add(Convolution1D(32, 3, border_mode="same",activation="relu",input_shape=(15000, 1)))
cnn.add(MaxPooling1D(pool_length=(2)))
cnn.add(Flatten())
cnn.add(Dense(32, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(1, activation="sigmoid"))

from sklearn.metrics import confusion_matrix
cnn.load_weights("logs/cnn1layer/checkpoint-07.hdf5")
y_pred = cnn.predict_classes(X_test)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
y_prob = cnn.predict_proba(X_test)
np.savetxt("cnn.txt", y_prob)


import os
for file in os.listdir("logs/cnn1layer/"):
  cnn.load_weights("logs/cnn1layer/"+file)
  y_pred = cnn.predict_classes(X_test)
  cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  loss, accuracy = cnn.evaluate(X_test, y_test)
  print(file)
  print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
  print("---------------------------------------------------------------------------------")
  accuracy = accuracy_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred , average="binary")
  precision = precision_score(y_test, y_pred , average="binary")
  f1 = f1_score(y_test, y_pred, average="binary")

  print("accuracy")
  print("%.3f" %accuracy)
  print("precision")
  print("%.3f" %precision)
  print("recall")
  print("%.3f" %recall)
  print("f1score")
  print("%.3f" %f1)

