import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)

traindata = pd.read_csv('data/train.csv', header=None)
testdata = pd.read_csv('data/test.csv', header=None)

X = traindata.iloc[:,1:61]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:61]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)


traindata = np.array(trainX)
trainlabel = np.array(Y)

testdata = np.array(testT)
testlabel = np.array(C)






print("-----------------------------------------LR---------------------------------")
model = LogisticRegression()
model.fit(traindata, trainlabel)

expected = testlabel
predicted = model.predict(testdata)

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)



print("-----------------------------------------NB---------------------------------")

model = GaussianNB()
model.fit(traindata, trainlabel)


expected = testlabel
predicted = model.predict(testdata)


y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")


print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


print("-----------------------------------------KNN---------------------------------")

model = KNeighborsClassifier()
model.fit(traindata, trainlabel)


expected = testlabel
predicted = model.predict(testdata)


y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")


print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


print("-----------------------------------------DT---------------------------------")

model = DecisionTreeClassifier()
model.fit(traindata, trainlabel)

expected = testlabel
predicted = model.predict(testdata)

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")


print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)


print("-----------------------------------------Adaboost---------------------------------")

model = AdaBoostClassifier(n_estimators=100)
model.fit(traindata, trainlabel)


expected = testlabel
predicted = model.predict(testdata)

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")


print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)



print("--------------------------------------RF--------------------------------------")
model = RandomForestClassifier(n_estimators=100)
model = model.fit(traindata, trainlabel)


expected = testlabel
predicted = model.predict(testdata)

y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")


print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)










'''
from sklearn import svm

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]

for C in C_2d_range:
    for gamma in gamma_2d_range:
      print("--------------------------------------------------")
      print("c")
      print(C)
      print("gamma")
      print(gamma)
      model = svm.SVC(kernel='linear', C=C, gamma=gamma) 

      model = model.fit(traindata, trainlabel)
      # make predictions
      expected = testlabel
      predicted = model.predict(testdata)
      # summarize the fit of the model
      #np.savetxt('res/RFpredicted.txt',predicted , fmt='%01d')
      print("--------------------------------------SVM----linear----------------------------------")

      y_train1 = expected
      y_pred = predicted
      accuracy = accuracy_score(y_train1, y_pred)
      recall = recall_score(y_train1, y_pred , average="binary")
      precision = precision_score(y_train1, y_pred , average="binary")
      f1 = f1_score(y_train1, y_pred, average="binary")

      print("----------------------------------------------")
      print("accuracy")
      print("%.3f" %accuracy)
      print("precision")
      print("%.3f" %precision)
      print("racall")
      print("%.3f" %recall)
      print("f1score")
      print("%.3f" %f1)


from sklearn import svm


C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]

for C in C_2d_range:
    for gamma in gamma_2d_range:

model = svm.SVC(kernel='rbf', C=1, gamma=1,probability=True)
model = model.fit(traindata, trainlabel)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
# summarize the fit of the model
np.savetxt('svmpred.txt',predicted)
print("--------------------------------------SVM------rbf--------------------------------")
      
#print("c")
#print(C)
#print("gamma")
#print(gamma)
y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("----------------------------------------------")
print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("racall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)
'''

