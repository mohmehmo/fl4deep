from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import numpy as np
# from Utils.utils import save_pkl, pack_train_config
from sklearn.model_selection import train_test_split

from umlaut import UmlautCallback

model = Sequential()  # two layers
model.add(Dense(4, kernel_initializer="glorot_uniform", input_dim=2))
model.add(Activation("sigmoid"))
model.add(Dense(1, kernel_initializer="glorot_uniform", input_dim=4))
model.add(Activation("sigmoid"))

# change l2 to clipnorm due to updated keras version
# sgd = SGD(l2=0.0, learning_rate=0.05, decay=1e-6, momentum=0.11, nesterov=True)
sgd = SGD(clipnorm=0.0, learning_rate=0.05, decay=1e-6, momentum=0.11, nesterov=True)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

print("begin to train")
list1 = [1, 1]
label1 = [0]
list2 = [1, 0]
label2 = [1]
list3 = [0, 0]
label3 = [0]
list4 = [0, 1]
label4 = [1]
train_data = np.array((list1, list2, list3, list4))  # four samples for epoch = 1000
label = np.array((label1, label2, label3, label4))
X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.2)

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

model.fit(X_train, y_train, epochs=1000, batch_size=4, verbose=1, shuffle=True,
            callbacks=[cb],
            validation_data=(X_test, y_test)
          )
list_test = [0, 1]
test = np.array((list_test, list1))
classes = model.predict(test)

print(classes)


