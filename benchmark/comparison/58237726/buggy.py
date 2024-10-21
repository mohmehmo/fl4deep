import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from math import ceil
from tensorflow.keras.layers import Input, Dense, Lambda, Conv1D, GlobalAveragePooling3D, Conv3D, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from umlaut import UmlautCallback

BATCH_SIZE = 12


def generate_fake_data():
    for j in range(1, 240 + 1):
        if j < 120:
            yield np.ones((6, 16, 16, 16, 3)), np.array([0., 1.])
        else:
            yield np.zeros((6, 16, 16, 16, 3)), np.array([1., 0.])


def make_tfdataset(for_training=True):
    dataset = tf.data.Dataset.from_generator(generator=lambda: generate_fake_data(),
                                             output_types=(tf.float32,
                                                           tf.float32),
                                             output_shapes=(tf.TensorShape([6, 16, 16, 16, 3]),
                                                            tf.TensorShape([2])))
    dataset = dataset.repeat()
    if for_training:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def create_model(in_shape=(6, 16, 16, 16, 3)):
    input_layer = Input(shape=in_shape)
    reshaped_input = Lambda(lambda x: K.reshape(x, (-1, *in_shape[1:])))(input_layer)
    conv3d_layer = Conv3D(filters=64, kernel_size=8, strides=(2, 2, 2), padding='same')(reshaped_input)
    relu_layer_1 = ReLU()(conv3d_layer)
    pooling_layer = GlobalAveragePooling3D()(relu_layer_1)
    reshape_layer_1 = Lambda(lambda x: K.reshape(x, (-1, in_shape[0] * 64)))(pooling_layer)
    expand_dims_layer = Lambda(lambda x: K.expand_dims(x, 1))(reshape_layer_1)
    conv1d_layer = Conv1D(filters=1, kernel_size=1)(expand_dims_layer)
    relu_layer_2 = ReLU()(conv1d_layer)
    reshape_layer_2 = Lambda(lambda x: K.squeeze(x, 1))(relu_layer_2)
    out = Dense(units=2, activation='softmax')(reshape_layer_2)
    return Model(inputs=[input_layer], outputs=[out])


train_ds = make_tfdataset(for_training=True)
clf_model = create_model(in_shape=(6, 16, 16, 16, 3))
clf_model.summary()

cb = UmlautCallback(
    clf_model,
    session_name='ea',
    offline=True,
)

clf_model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_crossentropy'])

train_ds = train_ds.take(10)
Y = np.concatenate([y for x, y in train_ds], axis=0)
X = np.concatenate([x for x, y in train_ds], axis=0)

history = clf_model.fit(X,Y,
                        epochs=5,
                        steps_per_epoch=ceil(240 / BATCH_SIZE),
                        verbose=1,
                        callbacks=[cb]
                        )