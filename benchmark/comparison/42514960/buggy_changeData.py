import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from umlaut import UmlautCallback

def buildMyNetwork(inputs, bottleNeck):
    inputLayer = Input(shape=(inputs,))
    autoencoder = Dense(inputs * 2, activation='tanh')(inputLayer)
    autoencoder = Dense(inputs * 2, activation='tanh')(autoencoder)
    autoencoder = Dense(bottleNeck, activation='tanh')(autoencoder)
    autoencoder = Dense(inputs * 2, activation='tanh')(autoencoder)
    autoencoder = Dense(inputs * 2, activation='tanh')(autoencoder)
    autoencoder = Dense(inputs, activation='tanh')(autoencoder)
    autoencoder = Model(inputs=inputLayer, outputs=autoencoder)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder


dataSize = 1000
variables = 6
data = np.zeros((dataSize, variables))
data[:, 0] = np.random.uniform(0, 0.5, size=dataSize)
data[:, 1] = np.random.uniform(0, 0.5, size=dataSize)
data[:, 2] = data[:, 0] + data[:, 1]
data[:, 3] = data[:, 0] * data[:, 1]
data[:, 4] = data[:, 0] / data[:, 1]
data[:, 5] = data[:, 0] ** data[:, 1]

trainData, testData = data[:900], data[900:]


model = buildMyNetwork(variables, 2)
model.fit(trainData, trainData, epochs=2000)
predictions = model.predict(testData)
