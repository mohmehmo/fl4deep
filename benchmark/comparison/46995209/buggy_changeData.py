from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

from umlaut import UmlautCallback

T = 1000
X = np.array(range(T))
Y = np.sin(3.5 * np.pi * X / T)

# fix 1: normalization X
X = (X - np.max(X)) / np.max(X)

X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.5)

input_dim = 1

model = Sequential()
model.add(Dense(10, input_dim=input_dim, activation='relu'))
model.add(Dense(25, input_dim=input_dim, activation='relu'))
model.add(Dense(1, activation='tanh'))

# update
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)


model.fit(X, Y, epochs=50, batch_size=10,
          callbacks=[cb],
          validation_data=(X_test, y_test)
          )
