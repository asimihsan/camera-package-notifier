#!/usr/bin/env python

import functools
import multiprocessing
import pathlib
import pickle
import sys
from typing import List, Tuple, Dict, Any

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
    return camera_event.get_camera_event_package_present_as_numpy_data(
        image_x_pixels=image_x_pixels, image_y_pixels=image_y_pixels
    )


def get_all_data(
    camera_events_path: pathlib.Path,
    image_x_pixels: int = 299,
    image_y_pixels: int = 299,
) -> Tuple[List[np.array], List[bool]]:
    camera_event_manager = CameraEventManager(camera_events_path)

    X_all: List[List[np.array]] = []
    y_all: List[bool] = []

    events: List[CameraEvent] = [
        event for event in camera_event_manager.get_annotated_events()
    ]
    data: List[Tuple[List[List[np.array]], bool]]
    function = functools.partial(
        get_data, image_x_pixels=image_x_pixels, image_y_pixels=image_y_pixels
    )
    with multiprocessing.Pool() as pool:
        data = pool.map(function, events)

    for datum in data:
        X_all.extend(datum[0])
        y_all.extend([datum[1]] * len(datum[0]))

    return (X_all, y_all)


def main(camera_events_path: pathlib.Path, output_data_path: pathlib.Path) -> None:
    image_x_pixels: int = 299
    image_y_pixels: int = 299

    X_all, y_all = get_all_data(
        camera_events_path, image_x_pixels=image_x_pixels, image_y_pixels=image_y_pixels
    )

    cctx = zstd.ZstdCompressor(level=3)
    with output_data_path.open("wb") as f_out:
        with cctx.stream_writer(f_out) as compressor:
            pickler = pickle.Pickler(compressor, protocol=4)
            pickler.dump((X_all, y_all))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
