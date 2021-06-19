import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
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
from keras.optimizers import Adam,SGD
from keras.layers.embeddings import Embedding
from sklearn.model_selection  import cross_val_predict
from keras.callbacks import ModelCheckpoint
import numpy
import pickle
from bayes_opt import BayesianOptimization
import params
import keras

positive=sio.loadmat('positive2601.mat')
negative=sio.loadmat('negative2601.mat')
data_p=positive['positive2601']
data_n=negative['negative2601']
data_p =data_p.reshape(280,51,51,1)
data_n =data_n.reshape(560,51,51,1)
X= np.vstack((data_p,data_n))
y = np.array([1] * 280 + [0] * 560)

seed = 8
numpy.random.seed(seed)
X_train, X_val,  y_train, y_val = train_test_split(X,  y, test_size=0.2,random_state=seed)
epochs = 90

HP_NUM_UNITS=hp.HParam('num_units', hp.Discrete([16,32,64]))
HP_NUM_batch_size=hp.HParam('batch_size', hp.Discrete([16,32,64]))
HP_DROPOUT=hp.HParam('dropout', hp.Discrete([0.1,0.3,0.5]))
HP_LEARNING_RATE= hp.HParam('learning_rate', hp.Discrete([0.001,0.005, 0.0001]))
HP_OPTIMIZER=hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_ACCURACY='accuracy'

log_dir ='./' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
with tf.summary.create_file_writer(log_dir).as_default():
  hp.hparams_config(
  hparams=
  [HP_NUM_UNITS, HP_NUM_batch_size,HP_DROPOUT, HP_OPTIMIZER, HP_LEARNING_RATE],
  metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )


def create_model(hparams):
    from keras import models
    from keras import layers
    import numpy as np
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(51, 51, 1)))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Flatten())
    model.add(Dense(hparams[HP_NUM_UNITS]))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(hparams[HP_NUM_UNITS]))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(1, activation='sigmoid'))
    batch_size = hparams[HP_NUM_batch_size]
    optimizer = hparams[HP_OPTIMIZER]
    learning_rate = hparams[HP_LEARNING_RATE]
    if optimizer == "adam":
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

    else:
        raise ValueError("unexpected optimizer name: %r" % (optimizer_name,))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train,
               y_train,
              epochs=90,
              batch_size=batch_size ,
              validation_data=(X_val, y_val),
              verbose=2, shuffle=True,
              callbacks = [
              tf.keras.callbacks.TensorBoard(log_dir),  # log metrics
              hp.KerasCallback(log_dir, hparams),  # log hparams

    ])
    return history.history['val_accuracy'][-1]
def run(run_dir, hparams):
 with tf.summary.create_file_writer(run_dir).as_default():
  hp.hparams(hparams) # record the values used in this trial
  accuracy = create_model(hparams)
  #converting to tf scalar
  accuracy= tf.reshape(tf.convert_to_tensor(accuracy), []).numpy()
  tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0
for num_units in HP_NUM_UNITS.domain.values:
 for dropout_rate in HP_DROPOUT.domain.values:
  for optimizer in HP_OPTIMIZER.domain.values:
    for learning_rate in HP_LEARNING_RATE.domain.values:
        for batch_size in HP_NUM_batch_size.domain.values:
            hparams = {
            HP_NUM_UNITS: num_units,
            HP_DROPOUT: dropout_rate,
            HP_OPTIMIZER: optimizer,
            HP_LEARNING_RATE: learning_rate,
            HP_NUM_batch_size:batch_size
             }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams)
            session_num += 1