#!/usr/bin/env python

import pathlib
import sys

from tensorflow import keras
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    from camera_event import CameraEvent, CameraEventManager
else:
    from .camera_event import CameraEvent, CameraEventManager


def main(model_path: pathlib.Path, data_path: pathlib.Path, event_id: str) -> None:

    camera_event_manager: CameraEventManager = CameraEventManager(
        data_path, images_subpath="images"
    )
    event: CameraEvent = camera_event_manager.get_event(event_id)
    (imgs, is_package_present) = event.get_camera_event_actual_images_as_numpy_data()

    model = keras.models.load_model(str(model_path.absolute()))

    inputs = [np.asarray([x]) for x in imgs]

    prediction: np.array = model.predict(x=inputs)
    print("prediction: %s" % (prediction[0][0],))
    print("actual: %s" % (is_package_present,))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3])
