from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from umlaut import UmlautCallback

RADIX = 7


def _get_number(vector):
    return sum(x * 2 ** i for i, x in enumerate(vector))


def _get_mod_result(vector):
    return _get_number(vector) % RADIX


def _number_to_vector(number):
    binary_string = bin(number)[2:]
    if len(binary_string) > 20:
        raise NotImplementedError
    bits = (((0,) * (20 - len(binary_string))) +
            tuple(map(int, binary_string)))[::-1]
    assert len(bits) == 20
    return np.c_[bits]


def get_mod_result_vector(vector):
    return _number_to_vector(_get_mod_result(vector))


X = np.random.randint(2, size=(10000, 20))
Y = np.vstack(map(get_mod_result_vector, X))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

model = Sequential()
model.add(Dense(units=20, activation='relu', input_dim=20))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=20, activation='softmax'))

# fix 1: opt 'sgd' -> 'Adam'
# fix 2: lr = 0.01 -> 0.0001
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

model.fit(x_train, y_train, epochs=10, batch_size=50,
          callbacks=[cb],
          validation_data=(x_test, y_test)
          )