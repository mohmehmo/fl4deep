from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from umlaut import UmlautCallback


df = pd.read_csv("./fl4ml/41600519/training.csv")

y = np.array(df['Label'].apply(lambda x: 0 if x=='s' else 1))
X = np.array(df.drop(["EventId","Label", "KaggleSet", "KaggleWeight", "Weight"], axis=1))
sm = SMOTE()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# X_res, y_res = sm.fit_resample(X_train, y_train)


model = Sequential()
model.add(Dense(25, input_shape=(30,),activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])

# model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=100,batch_size=100)

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

model.fit(X_train,y_train,epochs=100,batch_size=100,
          callbacks=[cb],
          validation_data=(X_test, y_test))