#!/usr/bin/env python

from typing import List
import logging
import math
import multiprocessing
import os
import pathlib
import shutil
import sys

from image_extractor import ImageExtractor

logger = logging.getLogger("extract_images")

logging.getLogger().setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logging.getLogger().addHandler(ch)


def process_directory(directory: pathlib.Path) -> None:
    logger.info("main - processing %s..." % (directory,))

    old_frames_dir: pathlib.Path = directory / "frames"
    if old_frames_dir.is_dir():
        logger.info("deleting deprecated frames directory %s", old_frames_dir)
        shutil.rmtree(old_frames_dir.absolute())

    video_path: pathlib.Path = directory / "video.mp4"
    destination_path: pathlib.Path = directory / "images"
    flag_file: pathlib.Path = directory / "extracted_images"
    if flag_file.is_file():
        logger.info("main - already processed %s images, skipping" % (directory,))
        return

    image_extractor: ImageExtractor = ImageExtractor(video_path, destination_path)
    image_extractor.extract_images()

    flag_file.touch()


def main(source_directory: pathlib.Path) -> None:
    children: List[pathlib.Path] = [
        child
        for child in source_directory.iterdir()
        if child.is_dir()
        and (child / "successful").is_file()
        and not (child / "extracted_images").is_file()
    ]
    children.sort()
    with multiprocessing.Pool(
        processes=max(math.floor(os.cpu_count() / 2), 1), maxtasksperchild=8
    ) as pool:
        pool.map(process_directory, children)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
