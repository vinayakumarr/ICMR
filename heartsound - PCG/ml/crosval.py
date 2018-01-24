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
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

with open('trainnormal.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

trainX = np.array(your_list)

traindata = pd.read_csv('trainlabels.csv', header=None)
Y = traindata.iloc[:,0]
y_train1 = np.array(Y)
y_train= to_categorical(y_train1)

maxlen = 4000
trainX = sequence.pad_sequences(trainX, maxlen=maxlen)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))


def create_model():
   print("----------------------------------")
   model = Sequential()
   model.add(GRU(512,input_dim=4000, return_sequences=True)) 
   model.add(Dropout(0.1))
   model.add(GRU(512,return_sequences=True))
   model.add(Dropout(0.1))
   model.add(GRU(512,return_sequences=True))
   model.add(Dropout(0.1))
   model.add(GRU(512,return_sequences=False))
   model.add(Dropout(0.1))
   model.add(Dense(3))
   model.add(Activation('softmax'))
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   return model

seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn=create_model, nb_epoch=50, batch_size=2)

# evaluate using 10-fold cross validation
kfold = StratifiedKFold(y=y_train1, n_folds=5, shuffle=True, random_state=seed)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print(results)
print("results mean")
print(results.mean())

