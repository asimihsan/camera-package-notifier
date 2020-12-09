#!/usr/bin/env python

"""

References:

[1] https://keras.io/guides/transfer_learning/
"""

from typing import List, Tuple
import pathlib
import sys

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Concatenate
import numpy as np

if __name__ == "__main__":
    from camera_event import CameraEvent, CameraEventManager
else:
    from .camera_event import CameraEvent, CameraEventManager


def create_model(
    input_shape: Tuple[int, int, int] = (299, 299, 3), number_of_input_images: int = 5
) -> keras.Model:
    base_model = Xception(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    )
    base_model.trainable = False

    inputs = []
    outputs = []
    for _ in range(number_of_input_images):
        input = keras.Input(shape=input_shape)
        inputs.append(input)

        # Pre-trained Xception weights requires that input be normalized
        # from (0, 255) to a range (-1., +1.), the normalization layer
        # does the following, outputs = (inputs - mean) / sqrt(var)
        norm_layer = layers.experimental.preprocessing.Normalization()
        mean = np.array([127.5] * 3)
        var = mean ** 2

        # Scale inputs to [-1, +1]
        input = norm_layer(input)
        norm_layer.set_weights([mean, var])

        output = base_model(input, training=False)
        flattened = Flatten()(output)
        outputs.append(flattened)

    output = Concatenate()(outputs)
    output = layers.Dropout(0.5)(output)
    output = layers.Dense(128, activation="relu")(output)
    output = layers.Dropout(0.5)(output)
    output = layers.Dense(64, activation="relu")(output)
    output = Dense(1)(output)
    model = keras.Model(inputs=inputs, outputs=output)
    model.summary()
    return model


def get_all_data(
    camera_events_path: pathlib.Path,
    image_x_pixels: int = 299,
    image_y_pixels: int = 299,
) -> Tuple[List[np.array], List[bool]]:
    camera_event_manager = CameraEventManager(camera_events_path)
    camera_events: List[CameraEvent] = [
        event for event in camera_event_manager.get_annotated_events()
    ]
    package_present_events: List[CameraEvent] = [
        event for event in camera_events if event.get_annotation_package_present()
    ]
    package_not_present_events: List[CameraEvent] = [
        event for event in camera_events if not event.get_annotation_package_present()
    ]

    X_all: List[List[np.array]] = []
    y_all: List[bool] = []

    for camera_event in package_present_events + package_not_present_events:
        data = camera_event.get_camera_event_package_present_as_numpy_data(
            image_x_pixels=image_x_pixels, image_y_pixels=image_y_pixels
        )
        X_all.extend(data[0])
        y_all.extend([data[1]] * len(data[0]))

    return (X_all, y_all)


def convert_X_to_keras_input(X: List[List[np.ndarray]]) -> List[np.array]:
    X_2 = []
    for i in range(len(X[0])):
        subelems = [x[i] for x in X]
        X_2.append(subelems)
    X_3 = [np.asarray(x) for x in X_2]
    return X_3


def main(camera_events_path: pathlib.Path) -> None:
    image_x_pixels: int = 299
    image_y_pixels: int = 299

    # TODO clean up how X_all is created
    # TODO cross-validation
    # TODO test hold-out
    X_all, y_all = get_all_data(
        camera_events_path, image_x_pixels=image_x_pixels, image_y_pixels=image_y_pixels
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, stratify=y_all, test_size=0.1, random_state=42
    )

    model: keras.Model = create_model(input_shape=(image_x_pixels, image_y_pixels, 3))
    loss_fn = keras.losses.Hinge()
    optimizer = keras.optimizers.Adam()

    model.compile(loss=loss_fn, optimizer=optimizer)

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3)
    model_checkpoint_callback = ModelCheckpoint(
        "/tmp/camera_model.h5", monitor="val_loss", verbose=0, save_best_only=True,
    )

    model.fit(
        x=convert_X_to_keras_input(X_train),
        y=np.asarray(y_train),
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
        class_weight={False: 0.5, True: 0.5},
        callbacks=[early_stopping_callback, model_checkpoint_callback],
    )

    results = model.evaluate(
        convert_X_to_keras_input(X_test), np.asarray(y_test), batch_size=32
    )
    print("test loss, test acc: ", results)

    import ipdb

    ipdb.set_trace()
    pass


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
