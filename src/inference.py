#!/usr/bin/env python

import pathlib
import sys

from tensorflow import keras
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

if __name__ == "__main__":
    from camera_event import CameraEvent, CameraEventManager
else:
    from .camera_event import CameraEvent, CameraEventManager


def get_f1(y_true: np.array, y_pred: np.array) -> float:
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val: float = 2 * (precision * recall) / (precision + recall + K.epsilon())
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


def main(model_path: pathlib.Path, data_path: pathlib.Path, event_id: str) -> None:

    camera_event_manager: CameraEventManager = CameraEventManager(
        data_path, images_subpath="images"
    )
    event: CameraEvent = camera_event_manager.get_event(event_id)
    (imgs, is_package_present) = event.get_camera_event_actual_images_as_numpy_data()

    # https://stackoverflow.com/questions/51700351/valueerror-unknown-metric-function-when-using-custom-metric-in-keras
    custom_objects = {
        "get_f1": get_f1,
        "matthews_correlation": matthews_correlation,
    }
    model = keras.models.load_model(
        str(model_path.absolute()), custom_objects=custom_objects
    )

    inputs = [np.asarray([x]) for x in imgs]

    prediction: np.array = model.predict(x=inputs)
    print("prediction: %s" % (prediction[0][0],))
    print("actual: %s" % (is_package_present,))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3])
