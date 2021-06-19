import numpy as np
from keras import regularizers
import matplotlib.pyplot as plt
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from itertools import chain
import scipy.io as sio
import random
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from sklearn.model_selection  import cross_val_predict
import math
import numpy

positive=sio.loadmat('positive2601.mat')
negative=sio.loadmat('negative2601.mat')
data_p=positive['positive2601']
data_n=negative['negative2601']
data_p =data_p.reshape(280,51,51,1)
data_n =data_n.reshape(560,51,51,1)
X= np.vstack((data_p,data_n))
y = np.array([1] * 280 + [0] * 560)
seed=8
numpy.random.seed(seed)

kf = StratifiedKFold(n_splits=5 ,shuffle=True, random_state=seed)
cvSn=[]
cvSp=[]
cvMCC=[]
cvscores=[]
for train_index, test_index in kf.split(X,y):#分层交叉验证，正集分k份，
    # 负集分k份
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = Sequential()
    model.add(Conv2D(66, (3, 3), padding='same', activation='relu', input_shape=(51,51,1)))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))
    model.add(Conv2D(98, (3, 3), padding='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(57,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(57, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(57, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(57, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.000694227),metrics=['accuracy'])
    #print(model.summary())

    history = model.fit(x_train, y_train,epochs=100, batch_size=89,validation_data=(x_test,y_test),
                        callbacks=[EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto',
                                                 restore_best_weights=True)
                                  ],
    verbose=2, shuffle=True)
    score = model.evaluate(x_test, y_test)
    y_predict=model.predict_classes(x_test)

    c = list(chain(*y_predict))


    def Twoclassfy_evalu(y_test, y_predict):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        FP_index = []
        FN_index = []
        for i in range(len(y_test)):
            if y_predict[i] > 0.5 and y_test[i] == 1:
                TP += 1
            if y_predict[i] > 0.5 and y_test[i] == 0:
                FP += 1
                FP_index.append(i)
            if y_predict[i] < 0.5 and y_test[i] == 1:
                FN += 1
                FN_index.append(i)
            if y_predict[i] < 0.5 and y_test[i] == 0:
                TN += 1
        Sn = TP / (TP + FN)
        Sp = TN / (FP + TN)
        MCC=(TP*TN-FP*FN)/math.sqrt((TN + FN)*(FP+TN)*(TP+FN)*(TP+FP))
        Acc = (TP + TN) / (TP + FP + TN + FN)
        cvSn.append(Sn * 100)
        cvSp.append(Sp * 100)
        cvMCC.append(MCC * 100)

        print('Sn', Sn)
        print('Sp', Sp)
        print('MCC', MCC)
        print('ACC', Acc)
        # Mcc = (TP * TN - FP * FN) / (math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))
        return Sn, Sp, Acc
    if __name__ == '__main__':
        Twoclassfy_evalu(y_test, c)

    cvscores.append(score[1] * 100)
    print(score)

print(cvSn)
print(cvSp)
print(cvMCC)
print(cvscores)
meanSn=np.mean(cvSn)
meanSp=np.mean(cvSp)
meanMCC=np.mean(cvMCC)
meanscores=np.mean(cvscores)
print(meanSn)
print(meanSp)
print(meanMCC)
print(meanscores)
