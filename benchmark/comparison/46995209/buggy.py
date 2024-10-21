from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy
from sklearn.model_selection import train_test_split

from umlaut import UmlautCallback

T = 1000
X = numpy.array(range(T))
Y = numpy.sin(3.5 * numpy.pi * X / T)
X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2)

input_dim = 1

model = Sequential()
model.add(Dense(10, input_dim=input_dim, activation='tanh'))
model.add(Dense(90, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='tanh'))


cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
model.fit(X, Y, epochs=50, batch_size=10,
          callbacks=[cb],
          validation_data=(X_test, y_test))