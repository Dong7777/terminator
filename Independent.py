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
from keras.models import load_model
from keras.models import load_model




positive=sio.loadmat('dc1472601.mat')
data_d=positive['dc1472601']
x_testone=data_d.reshape(147,51,51,1)
y_testone = np.array([1] * 147)

duliceshi=sio.loadmat('dc4252601')
data_dc=duliceshi['dc4252601']
x_testtwo =data_dc.reshape(425,51,51,1)
y_testtwo = np.array([1] * 425)

duliceshi=sio.loadmat('dc252601')
data_dc=duliceshi['dc252601']
x_testthree =data_dc.reshape(25,51,51,1)
y_testthree = np.array([1] * 25)

positive=sio.loadmat('dc159.mat')
data_d=positive['dc159']
x_testfour = data_d.reshape(159,51,51,1)
y_testfour = np.array([0] * 159)

positive=sio.loadmat('dc122.mat')
data_d=positive['dc122']
x_testfive = data_d.reshape(122,51,51,1)
y_testfive = np.array([0] * 122)


model = load_model('my_model1.h5')


scoreone = model.evaluate(x_testone, y_testone)
scoretwo = model.evaluate(x_testtwo, y_testtwo)
scorethree = model.evaluate(x_testthree, y_testthree)
scorefour = model.evaluate(x_testfour, y_testfour)
scorefive= model.evaluate(x_testfive, y_testfive)

print(scoreone)
print(scoretwo)
print(scorethree)
print(scorefour)
print(scorefive)



