from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras.utils import np_utils

from umlaut import UmlautCallback

# Load the iris dataset from seaborn.
iris = load_iris()

# Use the first 4 variables to predict the species.
X, y = iris.data[:, :4], iris.target

# Split both independent and dependent variables in half for cross-validation
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=0)


# Define a one-hot encoding of variables in an array.
def one_hot_encode_object_array(arr):
    # One hot encode a numpy array of objects (e.g. strings)
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

# One-hot encode the train and test y's
# train_y = one_hot_encode_object_array(train_y)
# test_y = one_hot_encode_object_array(test_y)

# Build the keras model

model = Sequential()
# 4 features in the input layer (the four flower measurements)
# 16 hidden units
model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))
# 3 classes in the ouput layer (corresponding to the 3 species)
model.add(Dense(3))
model.add(Activation('softmax'))
# fix 1: loss: categorical_crossentropy -> sparse_categorical_crossentropy
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

# Train the keras model
model.fit(train_X, train_y, verbose=1, batch_size=1, epochs=100,
            callbacks=[cb],
            validation_data=(test_X, test_y))
