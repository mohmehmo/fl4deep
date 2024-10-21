from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np
# from Utils.utils import save_pkl, pack_train_config
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
    # Y += [np.random.randint(1, 101) / 100]
    Y += [(count + 1) / 100]
X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2)
print(str(X) + ' ' + str(Y))

# create model
model = Sequential()
model.add(Dense(50, input_dim=1, kernel_initializer='uniform', activation='relu'))
model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model
opt = optimizers.SGD(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

# Fit the model
model.fit(X, Y, epochs=150, batch_size=100,
          callbacks=[cb]
          )

# evaluate the model
scores = model.evaluate(X, Y)
predictions = model.predict(X)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print(str(predictions))
