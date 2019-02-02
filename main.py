# Python 2.7 on Jupyter
# Libraries: Keras, pandas, numpy, matplotlib, seaborn

# For compatibility
from __future__ import absolute_import
from __future__ import print_function
# For manipulating data
import pandas as pd
import numpy as np
from keras.utils import np_utils # For y values
# For plotting
%matplotlib inline
import seaborn as sns
# For Keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout


# Set data
data = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [2, 2, 0],
    [3, 3, 0],
    [4, 4, 0],
    [5, 5, 0],
    [6, 6, 0],
    [7, 7, 0],
    [8, 8, 0],
    [9, 9, 0],
    [10, 10, 1],
    [11, 11, 1],
    [12, 12, 1],
    [13, 13, 1],
    [14, 14, 1],
    [15, 15, 1],
    [16, 16, 1],
    [17, 17, 1],
    [18, 18, 1],
    [19, 19, 1]
])


data = np.vstack((data, data, data, data)) # Just for sufficient input
data = pd.DataFrame(data, columns=['x', 'y', 'class'])

# Split X and y
X = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values

# Get dimensions of input and output
dimof_input = X.shape[1]
dimof_output = np.max(y) + 1
print('dimof_input: ', dimof_input)
print('dimof_output: ', dimof_output)

# Set y categorical
y = np_utils.to_categorical(y, dimof_output)

# Set constants
batch_size = 1528
dimof_middle = 100
dropout = 0.1
countof_epoch = 500
verbose = 0
print()
print('---Model Parameters---')
print('batch_size: ', batch_size)
print('dimof_middle: ', dimof_middle)
print('dropout: ', dropout)
print('countof_epoch: ', countof_epoch)
print('verbose: ', verbose)
print()

# Set model
model = Sequential()
model.add(Dense(dimof_middle, input_dim=dimof_input, kernel_initializer='uniform', activation='tanh'))
model.add(Dropout(dropout))
model.add(Dense(dimof_middle, kernel_initializer='uniform', activation='tanh'))
model.add(Dropout(dropout))
model.add(Dense(dimof_output, kernel_initializer='uniform', activation='softmax'))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# Train
model.fit(
    X, y,
    validation_split=0.2,
    batch_size=batch_size, nb_epoch=countof_epoch, verbose=verbose)

# Evaluate
loss, accuracy = model.evaluate(X, y, verbose=verbose)
print()
print('---Model Performance---')
print('loss: ', loss)
print('accuracy: ', accuracy)
print()

# Predict
# model.predict_classes(X, verbose=verbose)
print()
print('---Sample Predictions---')
print('prediction of [1, 1]: ', model.predict_classes(np.array([[1, 1]]), verbose=verbose))
print('prediction of [8, 8]: ', model.predict_classes(np.array([[8, 8]]), verbose=verbose))
print('prediction of [15, 1]: ', model.predict_classes(np.array([[15, 1]]), verbose=verbose))
print('prediction of [18, 8]: ', model.predict_classes(np.array([[18, 8]]), verbose=verbose))

# Plot
sns.lmplot('x', 'y', data, 'class', fit_reg=False).set(title='Data')
data_ = data.copy()
data_['class'] = model.predict_classes(X, verbose=0)
sns.lmplot('x', 'y', data_, 'class', fit_reg=False).set(title='Trained Result')
data_['class'] = [ 'Error' if is_error else 'Non Error' for is_error in data['class'] != data_['class']]
sns.lmplot('x', 'y', data_, 'class', fit_reg=False).set(title='Errors')
dimof_input:  2
dimof_output:  2
