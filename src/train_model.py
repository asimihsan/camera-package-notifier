#!/usr/bin/env python

"""

References:

[1] https://keras.io/guides/transfer_learning/
"""

from typing import List, Tuple
import pathlib
import pickle
import sys

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Concatenate
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications.densenet import preprocess_input
import zstandard as zstd

if __name__ == "__main__":
    from camera_event import CameraEvent, CameraEventManager
else:
    from .camera_event import CameraEvent, CameraEventManager


def create_model(
    input_shape: Tuple[int, int, int] = (299, 299, 3),
    number_of_input_images: int = 5,
    dropout: float = 0.2,
) -> keras.Model:
    base_model = Xception(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    )
    base_model.trainable = False
    preprocess_layer = tf.keras.applications.xception.preprocess_input

    xception_encoder_input = keras.Input(shape=(2048,))
    xception_encoder_internal = layers.Dropout(dropout)(xception_encoder_input)
    xception_encoder_internal = layers.Dense(256, activation="relu")(
        xception_encoder_internal
    )
    xception_encoder_internal = layers.Dropout(dropout)(xception_encoder_internal)
    xception_encoder_output = layers.Dense(128, activation="relu")(
        xception_encoder_internal
    )
    xception_encoder = keras.models.Model(
        xception_encoder_input, xception_encoder_output
    )

    inputs = []
    outputs = []
    for _ in range(number_of_input_images):
        input = keras.Input(shape=input_shape)
        inputs.append(input)

        input = preprocess_layer(input)
        output = base_model(input, training=False)
        output = Flatten()(output)
        output = xception_encoder(output)
        outputs.append(output)

    output = Concatenate()(outputs)
    output = layers.Dropout(dropout)(output)
    output = layers.Dense(128, activation="relu")(output)
    output = layers.Dropout(dropout)(output)
    output = layers.Dense(64, activation="relu")(output)
    output = Dense(1)(output)
    model = keras.Model(inputs=inputs, outputs=output)
    model.summary()
    return model


def convert_X_to_keras_input(X: List[List[np.ndarray]]) -> List[np.array]:
    X_2 = []
    for i in range(len(X[0])):
        subelems = [x[i] for x in X]
        X_2.append(subelems)
    X_3 = [np.asarray(x) for x in X_2]
    return X_3


def main(input_data_path: pathlib.Path, output_model_path: pathlib.Path) -> None:
    image_x_pixels: int = 299
    image_y_pixels: int = 299

    # TODO clean up how X_all is created
    # TODO cross-validation
    # TODO test hold-out
    print("loading data...")
    dctx = zstd.ZstdDecompressor()
    with input_data_path.open("rb") as f_in:
        with dctx.stream_reader(f_in) as reader:
            unpickler = pickle.Unpickler(reader)
            X_all, y_all = unpickler.load()
    print("loaded data.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, stratify=y_all, test_size=0.1, random_state=42
    )

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model: keras.Model = create_model(
            input_shape=(image_x_pixels, image_y_pixels, 3)
        )
        loss_fn = keras.losses.Hinge()

        # default learning rate is 1e-3
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        model.compile(loss=loss_fn, optimizer=optimizer)

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5)
    model_checkpoint_callback = ModelCheckpoint(
        str(output_model_path.absolute()),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
    )

    model.fit(
        x=convert_X_to_keras_input(X_train),
        y=np.asarray(y_train),
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        shuffle=True,
        class_weight={False: 0.5, True: 0.5},
        callbacks=[early_stopping_callback, model_checkpoint_callback],
    )

    results = model.evaluate(
        convert_X_to_keras_input(X_test), np.asarray(y_test), batch_size=32
    )
    print("test loss, test acc: ", results)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
