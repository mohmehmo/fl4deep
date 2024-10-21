import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from umlaut import UmlautCallback

cifar10 = tf.keras.datasets.cifar10.load_data()
(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10
label_dict = {0: 'airplane', 1: 'automobile', 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse",
              8: "ship", 9: "truck"}
train_images = x_img_train.astype('float32') / 255
test_images = x_img_test.astype('float32') / 255
train_labels = np_utils.to_categorical(y_label_train)
test_labels = np_utils.to_categorical(y_label_test)
model = Sequential()

model.add(Conv2D(filters=256,
                 kernel_size=(3, 3),
                 activation='relu',
                 data_format='channels_last',
                 input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=128,
                 kernel_size=(2, 2),
                 activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=categorical_crossentropy,
              metrics=['accuracy'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

#fix1
model.fit(train_images, train_labels,
          batch_size=500,
          epochs=15,
          verbose=1,
          validation_data=(test_images[:2000], test_labels[:2000]),
          callbacks=[cb]
          )

loss, accuracy = model.evaluate(test_images[:2000], test_labels[:2000])
print('loss: ', loss, '\naccuracy: ', accuracy)
