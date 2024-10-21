import numpy
import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from umlaut import UmlautCallback

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("./fl4ml/59278771/dataset/iris.data", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X, X_test, dummy_y, y_test = train_test_split(X, dummy_y, test_size=0.)

'''
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation="relu", kernel_initializer="normal"))
    model.add(Dense(3, activation="sigmoid", kernel_initializer="normal"))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
'''

# change to model.fit()

# create model
model = Sequential()
# fix 2: act relu -> selu
model.add(Dense(4, input_dim=4,
                # activation="selu",
                kernel_initializer="normal"))
# fix 3: act sigmoid -> softmax
model.add(Dense(3,
                # activation="softmax",
                kernel_initializer="normal"))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

model.fit(X, dummy_y, epochs=200, batch_size=5, verbose=1,
          callbacks=[cb],
          validation_data=(X_test, y_test)
          )

