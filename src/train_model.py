#!/usr/bin/env python

"""

References:

[1] https://keras.io/guides/transfer_learning/
"""

from typing import List, Tuple
import pathlib
import pickle
import sys

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Concatenate
from tensorflow.python.keras.applications.xception import Xception
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import zstandard as zstd

if __name__ == "__main__":
    from camera_event import CameraEvent, CameraEventManager
else:
    from .camera_event import CameraEvent, CameraEventManager


def create_model(
    input_shape: Tuple[int, int, int] = (299, 299, 3),
    number_of_input_images: int = 5,
    dropout: float = 0.5,
) -> keras.Model:
    # base_model = NASNetLarge(
    #    include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    # )
    # base_model.trainable = False
    # preprocess_layer = tf.keras.applications.nasnet.preprocess_input

    base_model = Xception(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    )
    base_model.trainable = False
    preprocess_layer = tf.keras.applications.xception.preprocess_input

    # NASNetLarge
    # base_model_encoder_input = keras.Input(shape=(4032,))

    # Xception is 2048
    base_model_encoder_input = keras.Input(shape=(2048,))

    base_model_encoder_internal = layers.Dropout(dropout)(base_model_encoder_input)
    base_model_encoder_output = layers.Dense(256, activation="elu")(
        base_model_encoder_internal
    )
    base_model_encoder = keras.models.Model(
        base_model_encoder_input, base_model_encoder_output
    )

    inputs = []
    outputs = []
    for _ in range(number_of_input_images):
        input = keras.Input(shape=input_shape)
        inputs.append(input)

        input = preprocess_layer(input)
        output = base_model(input, training=False)
        output = Flatten()(output)
        # output = base_model_encoder(output)
        outputs.append(output)

    output = Concatenate()(outputs)
    output = layers.Dropout(dropout)(output)
    output = layers.Dense(128, activation="elu")(output)
    output = layers.Dense(64, activation="elu")(output)
    output = Dense(1, activation="sigmoid")(output)
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


def get_f1(y_true: np.array, y_pred: np.array) -> float:
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val: float = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
def f1_loss(y_true: np.array, y_pred: np.array) -> float:
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    f1_val: float = 1 - K.mean(f1)
    return f1_val


def matthews_correlation(y_true: np.array, y_pred: np.array) -> float:
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = tp * tn - fp * fn
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    mcc: float = numerator / (denominator + K.epsilon())
    return mcc


def main(input_data_path: pathlib.Path, output_model_path: pathlib.Path) -> None:
    image_x_pixels: int = 299
    image_y_pixels: int = 299

    print("loading data...")
    dctx = zstd.ZstdDecompressor()
    with input_data_path.open("rb") as f_in:
        with dctx.stream_reader(f_in) as reader:
            unpickler = pickle.Unpickler(reader)
            (
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
            ) = unpickler.load()
    print("loaded data. X_train samples size: %d" % (len(X_train[0]),))

    # Convert class_weights to a dictionary to pass it to class_weight in model.fit
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(np.asarray(y_train)), y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model: keras.Model = create_model(
            input_shape=(image_x_pixels, image_y_pixels, 3)
        )

        # default learning rate is 1e-3
        optimizer = keras.optimizers.Adam(learning_rate=5e-4, amsgrad=True)

        model.compile(
            # loss=f1_loss,
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=optimizer,
            metrics=["accuracy", tf.keras.metrics.AUC(), get_f1, matthews_correlation],
        )

    early_stopping_callback = EarlyStopping(
        monitor="val_matthews_correlation", mode="max", patience=5
    )
    model_checkpoint_callback = ModelCheckpoint(
        str(output_model_path.absolute()),
        monitor="val_matthews_correlation",
        mode="max",
        verbose=0,
        save_best_only=True,
    )

    model.fit(
        x=X_train,
        y=y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_validation, y_validation),
        shuffle=True,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        class_weight=class_weights,
    )

    results = model.evaluate(X_test, y_test, batch_size=32)
    print("evaluation results: ", results)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
