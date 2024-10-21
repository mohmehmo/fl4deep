from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np

from sklearn.model_selection import train_test_split
from umlaut import UmlautCallback

# fix random seed for reproducibility
np.random.seed(7)

# split into input (X) and output (Y) variables
X = []
Y = []
count = 0

while count < 10000:
    count += 1
    X += [count / 10000]
    np.random.seed(count)
    Y += [(count + 1) / 100]

# fix 1: normalization
X = (X - np.max(X)) / np.max(X)
X = X.reshape(len(X),1)
# fix 2: normalization
Y = (Y - np.max(Y)) / np.max(Y)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

print(str(X) + ' ' + str(Y))

# create model
model = Sequential()
model.add(Dense(50, input_dim=1, kernel_initializer='uniform', activation='relu'))
model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model
# fix 3: optimizer SGD -> adam, fix 4: lr 0.01 -> 0.001
opt = optimizers.Adam(learning_rate=0.001)
# fix 5: loss 'binary_crossentropy' -> 'mse'
model.compile(loss='mse', optimizer=opt, metrics=['mse'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=100,
          callbacks=[cb],
          validation_data=(X_test, y_test)
          )

# evaluate the model
scores = model.evaluate(X, Y)
predictions = model.predict(X)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print(str(predictions))
