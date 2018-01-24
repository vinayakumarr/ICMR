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
from keras.layers import Convolution1D, GlobalMaxPooling1D,Dense, Dropout, Flatten, MaxPooling1D
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

print(trainX.shape)
# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))

batch_size = 2

model = Sequential()
model.add(Convolution1D(128, 3, border_mode="same",activation="relu",input_shape=(44100, 1)))
model.add(MaxPooling1D(pool_size=(2)))
#model.add(Convolution1D(256, 6, border_mode="same",activation="relu"))
#model.add(MaxPooling1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="logs/cnnlayer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=50, callbacks=[checkpointer])
model.save("logs/cnnlayer/lstm1layer_model.hdf5") 
