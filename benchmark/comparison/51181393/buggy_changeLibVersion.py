from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import linspace

from sklearn.model_selection import train_test_split
from umlaut import UmlautCallback

# Generate dummy data
data = data = linspace(1,2,100).reshape(-1,1)
y = data*5

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)

# Define the model
def baseline_model():
    model = Sequential()
    model.add(Dense(1, activation = 'linear', input_dim = 1))
    model.compile(optimizer=optimizers.RMSprop(lr=0.1), loss='mean_squared_error', metrics=['mae'])
    return model


# Use the model
regr = baseline_model()

cb = UmlautCallback(
    regr,
    session_name='ea',
    offline=True,
)

# regr.fit(data,y,epochs =200,batch_size = 32)

regr.fit(X_train,y_train,epochs =200,batch_size = 32,
         callbacks=[cb],
         validation_data=(X_test, y_test)
         )