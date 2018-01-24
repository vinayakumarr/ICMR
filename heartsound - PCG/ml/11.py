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
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize

with open('trainnormal.csv', 'rb') as f:
    reader = csv.reader(f)
    #print(reader)
    your_list = list(reader)
    #print(your_list)

trainX = np.array(your_list)
#normalized_metrics = normalize(trainX, axis=0, norm='l1')
#print(normalized_metrics.shape)

traindata = pd.read_csv('trainlabels.csv', header=None)
Y = traindata.iloc[:,0]
y_train1 = np.array(Y)
y_train= to_categorical(y_train1)

maxlen = 100000


trainX = sequence.pad_sequences(trainX, maxlen=maxlen)
# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))


cnn = Sequential()
cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(100000, 1)))
cnn.add(Convolution1D(64, 3, border_mode="same", activation="relu"))
cnn.add(MaxPooling1D(pool_length=(2)))

cnn.add(Convolution1D(256, 3, border_mode="same", activation="relu"))
cnn.add(Convolution1D(256, 3, border_mode="same", activation="relu"))
cnn.add(MaxPooling1D(pool_length=(2)))
'''
cnn.add(Convolution1D(1024, 3, border_mode="same", activation="relu"))
cnn.add(Convolution1D(1024, 3, border_mode="same", activation="relu"))
cnn.add(MaxPooling1D(pool_length=(2)))
'''
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.5))
cnn.add(Dense(3, activation="softmax"))

# define optimizer and objective, compile cnn

cnn.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

# train
checkpointer = callbacks.ModelCheckpoint(filepath="results/cnn2results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('results/cnn2results/cnntrainanalysis2.csv',separator=',', append=False)
cnn.fit(X_train, y_train, nb_epoch=25, show_accuracy=True,validation_data=(X_train, y_train),callbacks=[checkpointer,csv_logger])





