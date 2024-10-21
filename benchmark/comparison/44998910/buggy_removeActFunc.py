import numpy as np

# data generation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation

from umlaut import UmlautCallback

x_train=np.random.rand(9000)
y_train=x_train**4+x_train**3-x_train
x_train=x_train.reshape(len(x_train),1)

x_test=np.linspace(0,1,100)
y_test=x_test**4+x_test**3-x_test
x_test=x_test.reshape(len(x_test),1)

# model construction
model = Sequential()
model.add(Dense(units=200, input_dim=1))
# model.add(Activation('relu'))
model.add(Dense(units=45))
# model.add(Activation('relu'))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='sgd')

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

model.fit(x_train, y_train, epochs=40, batch_size=50, verbose=1,
            callbacks=[cb],
            validation_data=(x_test, y_test)
          )

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=100)