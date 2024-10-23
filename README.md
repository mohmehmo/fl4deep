# FL4Deep
A system-level fault localization approach for DL-based systems
***
### Repository structure
1. __benchmark:__ this directory consists of datasets that are used for training and validation of **_FL4Deep_**.
2. __sample_scripts:__ this dicrectory includes some example usage of **_FL4Deep_** within DL codes.
```
├── Analyzer
│    └── trace_training_log.py
├── Classifier
│    ├── models
│    └── classifier.py
├── benchmark
│    ├── comparison
│    └── training
├── kb
│    ├── model
│    ├── kb.b3
│    ├── kb_rootCauses.b3
│    ├── kb_rules.b3
│    └── rules.n3
├── logs
│    └── tmp_logs
├── sample_scripts
├── requirements.txt
└── README.md 
```

***
### How to use **_FL4Deep_**
In this version of **_FL4Deep_**, we use `Euler Yet another proof Engine (EYE)` (available [here](https://github.com/eyereasoner/eye)) as the reasoning engine.
After installing `EYE`, you need to clone this reposotiry to have **_FL4Deep_** souce code. Then, you need to call two **_FL4Deep_** APIs within the DL code including `data_analysis` to extract required information regarding dataset and `fl4ml` as a keras callback to collect information regarding model, training and deployment environment. Here is an example usage of **_FL4Deep_** in a DL code for MNIST problem. 

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from fl4ml_callback import fl4ml, data_analysis

def main_fl4ml():

    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    data_analysis(train=x_train, target=y_train, test=x_test)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 10

    model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                callbacks=[
                    fl4ml(batch = batch_size, epochs = epochs ,data=[x_train, y_train, x_test, y_test])
                         ])

if __name__ == "__main__":
    main_fl4ml()
```

***
### Prerequisites
``` python
Python       3.8 
keras        2.8.0
tensorflow   2.8.0
numpy        1.23.3
pandas       2.1.1
scikit-learn 0.23.1
rdflib       6.3.2
```