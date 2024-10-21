import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from math import ceil
from tensorflow.keras.layers import Input, Dense, Lambda, Conv1D, GlobalAveragePooling3D, Conv3D, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from umlaut import UmlautCallback

BATCH_SIZE = 6
INPUT_SHAPE = (16, 16, 16, 3)
BATCH_SHAPE = (BATCH_SIZE, *INPUT_SHAPE)


def generate_fake_data():
    for j in range(1, 240 + 1):
        if j < 120:
            yield np.ones(INPUT_SHAPE), np.array([0., 1.])
        else:
            yield np.zeros(INPUT_SHAPE), np.array([1., 0.])


def make_tfdataset(for_training=True):
    dataset = tf.data.Dataset.from_generator(generator=lambda: generate_fake_data(),
                                             output_types=(tf.float32,
                                                           tf.float32),
                                             output_shapes=(tf.TensorShape(INPUT_SHAPE),
                                                            tf.TensorShape([2])))
    dataset = dataset.repeat()
    if for_training:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def create_model(batch_size, input_shape):
    ipt = Input(batch_shape=(batch_size, *input_shape))
    x = Conv3D(filters=64, kernel_size=8, strides=(2, 2, 2),
               # activation='relu',
               padding='same')(ipt)
    x = Conv3D(filters=8, kernel_size=4, strides=(2, 2, 2),
               # activation='relu',
               padding='same')(x)
    x = GlobalAveragePooling3D()(x)
    out = Dense(units=2,
                # activation='softmax'
                )(x)
    return Model(inputs=ipt, outputs=out)


train_ds = make_tfdataset(for_training=True)

test_ds = make_tfdataset(for_training = False)
clf_model = create_model(BATCH_SIZE, INPUT_SHAPE)
clf_model.summary()
clf_model.compile(optimizer=Adam(learning_rate=1e-2),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_crossentropy'])
cb = UmlautCallback(
    clf_model,
    session_name='ea',
    offline=True,
)

history = clf_model.fit(train_ds,
                        epochs=5,
                        steps_per_epoch=ceil(240 / BATCH_SIZE),
                        verbose=1,
                        callbacks=[cb],
                        validation_data=test_ds
                        )
