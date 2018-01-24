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


testdata = pd.read_csv('data/testing_data_ann_spo2_60.csv', header=None)



C = testdata.iloc[:,0]
T = testdata.iloc[:,1:61]

scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


y_test = np.array(C)


# reshape input to be [samples, time steps, features]

X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


batch_size = 16

# 1. define the network
model = Sequential()
model.add(LSTM(32,input_dim=60))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.load_weights("logs/lstm3/checkpoint-923.hdf5")

y_pred = model.predict_classes(X_test)
y_prob = model.predict_proba(X_test)
np.savetxt("roc/32exact.txt", y_test, fmt="%01d")
np.savetxt("roc/32pred.txt", y_prob)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred , average="binary")
precision = precision_score(y_test, y_pred , average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

#np.savetxt("roc/32exact.txt", y_test, fmt="%01d")
#np.savetxt("roc/32pred.txt", y_prob)

print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("racall")
print("%.3f" %recall)
print("precision")
print("%.3f" %precision)
print("f1score")
print("%.3f" %f1)
#cm = metrics.confusion_matrix(y_test, y_prob)
print("==============================================")

'''
import os
for file in os.listdir("logs/lstm3/"):
  model.load_weights("logs/lstm3/"+file)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  loss, accuracy = model.evaluate(X_test, y_test)
  print(file)
  print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
'''
