import random
from keras.utils import np_utils
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from umlaut import UmlautCallback


batch_size = 32
nb_classes = 10
# fix 1: nb_epochs 2 -> 10
nb_epochs = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# mimic the noise described by author to reproduce similar poor result: loss: 0.6780 - val_loss: 0.6944
for i in range(len(y_train)):
    if y_train[i] < 4:
        y_train[i] = random.sample([1, 2, 3, 4], 1)[0]

for i in range(len(y_test)):
    if y_test[i] < 4:
        y_test[i] = random.sample([1, 2, 3, 4], 1)[0]

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Dense(units=100, input_shape = (784, ), activation="relu"))
model.add(Dense(units=200, activation="relu"))
model.add(Dense(units=200, activation="relu"))
model.add(Dense(units=nb_classes, activation="softmax"))

model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1,
            callbacks=[cb],
            validation_data=(X_test, y_test))