from keras.datasets import mnist
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import datetime
from keras.utils import np_utils

from umlaut import UmlautCallback

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# import data (New added)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(len(x_train), -1)
y_train = np_utils.to_categorical(y_train, 10)

model = Sequential()

bias_initializer = keras.initializers.Constant(value=0.1)

neurons_nb_layer_1 = 32
neurons_nb_layer_2 = 64
neurons_nb_layer_3 = 1024

model.add(Reshape((28, 28, 1), input_shape=(784,)))
model.add(Conv2D(filters=neurons_nb_layer_1, kernel_size=5 * 5, padding='same', activation="relu", bias_initializer=bias_initializer))
model.add(MaxPooling2D(padding='same'))
model.add(Conv2D(filters=neurons_nb_layer_2, kernel_size=5 * 5, padding='same', activation="relu", bias_initializer=bias_initializer))
model.add(MaxPooling2D(padding='same'))
model.add(Reshape((1, 7 * 7 * neurons_nb_layer_2)))
model.add(Dense(units=neurons_nb_layer_3, activation="relu", bias_initializer=bias_initializer))
model.add(Dropout(rate=0.5))
model.add(Flatten())
# fix 1: activation relu -> softmax
model.add(Dense(units=10, activation="softmax"))

model.summary()

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

start2 = datetime.datetime.now()
# fix 2: epoch 1 -> 2
'''
for i in range(1000):
    batch = mnist.train.next_batch(200)
    if i % 100 == 0:
        train_accuracy = model.evaluate(batch[0], batch[1])
        print("step", i, ":", train_accuracy)
    model.train_on_batch(batch[0], batch[1])
train_accuracy = model.evaluate(mnist.test.images, mnist.test.labels)
print("Accuracy", train_accuracy)
'''
# change to 'model.fit()'
model.fit(x_train, y_train, epochs=2, batch_size=50, verbose=1,
            callbacks=[cb],
            validation_data=(x_test, y_test)
          )
end2 = datetime.datetime.now()
time2 = (end2 - start2).seconds
print(time2 // 60, "min", time2 % 60, "s")
