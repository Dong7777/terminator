
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib as mpl
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,AveragePooling2D
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_objective
from skopt.utils import use_named_args
import os
import datetime
import scipy.io as sio
starttime = datetime.datetime.now()

positive=sio.loadmat('positive2601.mat')
negative=sio.loadmat('negative2601.mat')
data_p=positive['positive2601']
data_n=negative['negative2601']
data_p =data_p.reshape(280,51,51,1)
data_n =data_n.reshape(560,51,51,1)
X= np.vstack((data_p,data_n))
y = np.array([1] * 280 + [0] * 560)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

dim_learning_rate = Real(low=0.0001, high=0.01, prior='log-uniform',
                         name='learning_rate')

dim_num_dense_layers = Integer(low=1, high=10, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=5, high=128, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')
dim_filter1=Integer(low=8, high=128, name='filter1')
dim_filter2=Integer(low=8, high=128, name='filter2')
dim_droupt= Real(low=0.1, high=0.9, name='droupt')
dim_batch_size=Integer(low=8, high=128, name='batch_size')


dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation,
              dim_filter1,
              dim_filter2,
              dim_droupt,
              dim_batch_size
              ]
default_parameters = [0.001, 2, 32, 'relu',32,32,0.3,64]
# This is a function to log traning progress so that can be viewed by TnesorBoard.

def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, activation,filter1,filter2,droupt,batch_size):
    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation,filter1,filter2,droupt,batch_size)
    log_dir=os.path.join('log_dir')

    return log_dir


def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes, activation,filter1,filter2,droupt):
    model = Sequential()
    model.add(Conv2D(filter1, (3, 3), padding='same', activation=activation, input_shape=(51, 51, 1)))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Dropout(droupt))
    model.add(Conv2D(filter2, (3, 3), padding='same', activation=activation))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Dropout(droupt))
    model.add(keras.layers.Flatten())
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i + 1)

        # add dense layer
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name))
        model.add(Dropout(droupt))

    # use softmax-activation for classification.
    model.add(Dense(1, activation='sigmoid'))

    # Use the Adam method for training the network.
    optimizer = Adam(lr=learning_rate)

    # compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

path_best_model = '19_best_model.h5'
best_accuracy = 0.0
validation_data = (X_test, y_test)


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers,
            num_dense_nodes, activation,filter1,filter2,droupt,batch_size):
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print('filter1:', filter1)
    print('filter2:', filter2)
    print('droupt:', droupt)
    print('batch_size:', batch_size)
    print()
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation,
                         filter1=filter1,
                         filter2=filter2,
                         droupt=droupt)

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_dense_nodes, activation,filter1,filter2,droupt,batch_size)

    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False,
    )

    # Use Keras to train the model.
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=200,
                        batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks =[EarlyStopping(monitor='val_accuracy', patience=50, verbose=0, mode='auto',restore_best_weights=True),
                                   callback_log],verbose=2)

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = model.evaluate(x_val, y_val)

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)

        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()

    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy
# This function exactly comes from :Hvass-Labs, TensorFlow-Tutorials
fitness(x= default_parameters)


search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=50,
                            x0=default_parameters)
# you may examine other kind of minimization
# search_result = forest_minimize(func=fitness,
#                             dimensions=dimensions,
#                             base_estimator='RF',
#                             n_calls=50,
#                             random_state=4)
plot_convergence(search_result)
plt.savefig("1.png", dpi=400)

np.save('search_result',search_result)
np.save('search_result.x',search_result.x)
np.save('search_result.fun',search_result.fun)
np.save('sorted',sorted(zip(search_result.func_vals, search_result.x_iters)))

from skopt.plots import plot_objective_2D
fig = plot_objective_2D(result=search_result,
                        dimension_identifier1='learning_rate',
                        dimension_identifier2='num_dense_nodes',
                        levels=50)
plt.savefig("Lr_numnods.png", dpi=400)



endtime = datetime.datetime.now()
a=(endtime - starttime).seconds
print (a)