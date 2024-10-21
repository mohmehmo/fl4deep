import tensorflow as tf
from tensorflow.keras import layers
import keras
import numpy as np
import pandas as pd


from umlaut import UmlautCallback

batch_size = 32
epochs = 10
alpha = 0.0001
lambda_ = 0
h1 = 50

train = pd.read_csv('./fl4ml/55328966/dataset/mnist_train.csv.zip')
test = pd.read_csv('./fl4ml/55328966/dataset/mnist_test.csv.zip')

train = train.loc['1':'5000', :]
test = test.loc['1':'2000', :]

train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

x_train = train.loc[:, '1x1':'28x28']
y_train = train.loc[:, 'label']

x_test = test.loc[:, '1x1':'28x28']
y_test = test.loc[:, 'label']

x_train = x_train.values
y_train = y_train.values
x_test = x_test.values
y_test = y_test.values

nb_classes = 10
targets = y_train.reshape(-1)
y_train_onehot = np.eye(nb_classes)[targets]

nb_classes = 10
targets = y_test.reshape(-1)
y_test_onehot = np.eye(nb_classes)[targets]

model = tf.keras.Sequential()
model.add(layers.Dense(784, input_shape=(784,)))
model.add(layers.Dense(h1, activation='relu', kernel_regularizer=keras.regularizers.l2(lambda_)))
model.add(layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(lambda_)))

model.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(alpha), loss='binary_crossentropy', metrics=['accuracy'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

model.fit(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size,
          callbacks=[cb],
          validation_data=(x_test, y_test)
          )
