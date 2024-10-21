from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
import numpy

from sklearn.model_selection import train_test_split
from umlaut import UmlautCallback

X = numpy.array([[1., 1.], [0., 0.], [1., 0.], [0., 1.], [1., 1.], [0., 0.]])
y = numpy.array([[0.], [0.], [1.], [1.], [0.], [0.]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(2, input_dim=2, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(3, kernel_initializer='uniform', activation='sigmoid'))
#softmax cannot be used for 1-D vector
model.add(Dense(1, kernel_initializer='uniform', activation='softmax'))
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

history = model.fit(X_train, y_train, epochs=20,
                callbacks=[cb],
                validation_data=(X_test, y_test)
                )
