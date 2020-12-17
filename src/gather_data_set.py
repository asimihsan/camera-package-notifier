#!/usr/bin/env python

from typing import List, Tuple, Dict, Any
import functools
import math
import multiprocessing
import pathlib
import pickle
import random
import sys

from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
import numpy as np
import zstandard as zstd

if __name__ == "__main__":
    from camera_event import CameraEvent, CameraEventManager
else:
    from .camera_event import CameraEvent, CameraEventManager


def get_data(
    camera_event: CameraEvent, image_x_pixels: int = 299, image_y_pixels: int = 299,
) -> Tuple[List[List[np.array]], bool]:
    print("gathering %s..." % (camera_event,))
    data: Tuple[List[List[np.array]], bool]
    if camera_event.get_precomputed_data_flag_path().is_file():
        print("precomputed data already exists for %s" % (camera_event.root_path,))
        data = camera_event.load_precomputed_data()
        return data

    data = camera_event.get_camera_event_package_present_as_numpy_data(
        image_x_pixels=image_x_pixels, image_y_pixels=image_y_pixels
    )
    camera_event.save_precomputed_data(data)
    return data


def get_all_data(
    camera_events_path: pathlib.Path,
    image_x_pixels: int = 299,
    image_y_pixels: int = 299,
) -> Tuple[
    List[np.array], List[np.array], List[np.array], List[bool], List[bool], List[bool]
]:
    camera_event_manager = CameraEventManager(camera_events_path)

    events: List[CameraEvent] = [
        event for event in camera_event_manager.get_annotated_events()
    ]
    print("number of events: %d" % (len(events),))

    data: List[Tuple[List[List[np.array]], bool]]
    function = functools.partial(
        get_data, image_x_pixels=image_x_pixels, image_y_pixels=image_y_pixels
    )
    with multiprocessing.Pool() as pool:
        data = pool.map(function, events)

    X = [x[0] for x in data]
    y = [x[1] for x in data]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.1, shuffle=True, random_state=1
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.1, shuffle=True, random_state=2
    )

    X_train, y_train = unchunk_shuffle_augmented_block(X_train, y_train)
    X_validation, y_validation = unchunk_shuffle_augmented_block(
        X_validation, y_validation
    )
    X_test, y_test = unchunk_shuffle_augmented_block(X_test, y_test)

    return (X_train, X_validation, X_test, y_train, y_validation, y_test)


def unchunk_shuffle_augmented_block(
    X: List[List[List[np.array]]], y: List[bool]
) -> Tuple[List[np.array], List[bool]]:
    X_return: List[np.array] = []
    y_return: List[bool] = []
    for x_elem, y_elem in zip(X, y):
        X_return.extend(x_elem)
        y_return.extend([y_elem] * len(x_elem))
    X_return, y_return = shuffle(X_return, y_return)
    return X_return, y_return


def convert_X_to_keras_input(X: List[List[np.ndarray]]) -> List[np.array]:
    X_2 = []
    for i in range(len(X[0])):
        subelems = [x[i] for x in X]
        X_2.append(subelems)
    X_3 = [np.asarray(x) for x in X_2]
    return X_3


def main(camera_events_path: pathlib.Path, output_data_path: pathlib.Path) -> None:
    image_x_pixels: int = 299
    image_y_pixels: int = 299

    random.seed(42)
    np.random.seed(42)

    X_train, X_validation, X_test, y_train, y_validation, y_test = get_all_data(
        camera_events_path, image_x_pixels=image_x_pixels, image_y_pixels=image_y_pixels
    )
    X_train = convert_X_to_keras_input(X_train)
    X_validation = convert_X_to_keras_input(X_validation)
    X_test = convert_X_to_keras_input(X_test)
    y_train = np.asarray(y_train)
    y_validation = np.asarray(y_validation)
    y_test = np.asarray(y_test)

    print("saving to file...")
    cctx = zstd.ZstdCompressor(level=3, threads=-1)
    with output_data_path.open("wb") as f_out:
        with cctx.stream_writer(f_out) as compressor:
            pickler = pickle.Pickler(compressor, protocol=4)
            pickler.dump((X_train, X_validation, X_test, y_train, y_validation, y_test))
    print("saved to file.")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
