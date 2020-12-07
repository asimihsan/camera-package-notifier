#!/usr/bin/env python

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import math
import multiprocessing
import numpy as np
import pathlib
import random
import shutil
import skimage.transform
import sys

if __name__ == "__main__":
    from camera_event import CameraEvent, CameraEventManager
else:
    from .camera_event import CameraEvent, CameraEventManager


def change_contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    img2[:, :, 0] = cv2.multiply(img[:, :, 0], np.array([alpha]))
    img2 = cv2.cvtColor(img2, cv2.COLOR_Lab2RGB)
    return img2


def rotate_image(img: np.ndarray, angle_degrees: float) -> np.ndarray:
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_degrees, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.BORDER_REPLICATE)
    return result


def translate_image(
    img: np.ndarray, horizontal_proportion: float, vertical_proportion: float
) -> np.ndarray:
    num_rows, num_cols = img.shape[:2]

    translation_matrix = np.float32(
        [
            [1, 0, num_cols * horizontal_proportion],
            [0, 1, num_rows * vertical_proportion],
        ]
    )
    img2 = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    return img2


def augment(
    img: np.ndarray,
    using_change_contrast: bool,
    contrast_alpha: float,
    vertical_flip: bool,
    should_rotate: bool,
    rotate_degrees: float,
    should_translate: bool,
    translation_horizontal_proportion: float,
    translation_vertical_proportion: float,
) -> np.ndarray:
    new_img = np.copy(img)
    if using_change_contrast:
        new_img = change_contrast(new_img, contrast_alpha)
    if vertical_flip:
        new_img = cv2.flip(new_img, 1)
    if should_translate:
        new_img = translate_image(
            new_img,
            horizontal_proportion=translation_horizontal_proportion,
            vertical_proportion=translation_vertical_proportion,
        )
    if should_rotate:
        new_img = rotate_image(new_img, rotate_degrees)
    return new_img


def augment_images(
    camera_event: CameraEvent, number_of_augmentations: int = 20
) -> None:
    print("augmenting %s..." % (camera_event.root_path,))
    random.seed(camera_event.root_path.__hash__() % 2 ** 31)
    np.random.seed(camera_event.root_path.__hash__() % 2 ** 31)
    images_subpath: pathlib.Path = camera_event.images_subpath
    augmented_images_dirname: str = "%s_augmented" % (images_subpath.stem,)
    augmented_images_subpath: pathlib.Path = camera_event.root_path / augmented_images_dirname
    if augmented_images_subpath.is_dir():
        shutil.rmtree(str(augmented_images_subpath.absolute()))
    augmented_images_subpath.mkdir()
    for i in range(number_of_augmentations):
        destination_directory = augmented_images_subpath / f"{i:05d}"
        destination_directory.mkdir()
        generate_augmented_images(camera_event.get_image_paths(), destination_directory)


def generate_augmented_images(
    source_images: List[pathlib.Path], destination_directory: pathlib.Path
) -> None:
    using_change_contrast: bool = random.choice([True, False])
    contrast_alpha: float = random.uniform(0.5, 2.0)
    vertical_flip: bool = False
    should_rotate: bool = random.choice([True, False])
    rotate_degrees: int = random.randrange(-36, 36)
    should_translate: bool = random.choice([True, False])
    translation_horizontal_proportion: float = random.uniform(-0.1, 0.1)
    translation_vertical_proportion: float = random.uniform(-0.1, 0.1)
    for source_image in source_images:
        img = cv2.imread(str(source_image.absolute()))
        img2 = augment(
            img,
            using_change_contrast,
            contrast_alpha,
            vertical_flip,
            should_rotate,
            rotate_degrees,
            should_translate,
            translation_horizontal_proportion,
            translation_vertical_proportion,
        )
        destination_path = destination_directory / source_image.parts[-1]
        cv2.imwrite(str(destination_path.absolute()), img2)


def main(camera_events_path: pathlib.Path) -> None:
    camera_event_manager = CameraEventManager(camera_events_path)
    camera_events: List[CameraEvent] = [
        event for event in camera_event_manager.get_annotated_events()
    ]
    with multiprocessing.Pool() as pool:
        pool.map(augment_images, camera_events)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
