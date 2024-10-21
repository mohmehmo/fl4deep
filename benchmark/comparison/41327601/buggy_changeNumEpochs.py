# from Utils.utils import save_pkl, pack_train_config
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

from umlaut import UmlautCallback

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# input image dimensions
img_rows, img_cols = 28, 28

# reshape
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)

# model construction
model = Sequential()
model.add(Conv2D(100, 5, 5, padding='valid', input_shape=(28, 28, 1),
                        kernel_initializer='glorot_uniform',
                        activation='relu'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(100, 3, 3, kernel_initializer='glorot_uniform', activation='relu'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, kernel_initializer='glorot_uniform', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['categorical_accuracy'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

model.fit(x_train, y_train, epochs=1,
          callbacks=[cb],
          validation_data=(x_test, y_test))
