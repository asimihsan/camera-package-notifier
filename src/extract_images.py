#!/usr/bin/env python

import logging
import pathlib
import sys
import shutil
from typing import List

from image_extractor import ImageExtractor

logger = logging.getLogger("extract_images")

logging.getLogger().setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logging.getLogger().addHandler(ch)


def main(source_directory: pathlib.Path) -> None:
    children: List[pathlib.Path] = sorted(child for child in source_directory.iterdir())
    for child in children:
        if not child.is_dir():
            continue

        flag_file: pathlib.Path = child / "successful"
        if not flag_file.is_file():
            continue

        logger.info("main - processing %s..." % (child,))

        old_frames_dir: pathlib.Path = child / "frames"
        if old_frames_dir.is_dir():
            logger.info("deleting deprecated frames directory %s", old_frames_dir)
            shutil.rmtree(old_frames_dir.absolute())

        video_path: pathlib.Path = child / "video.mp4"
        destination_path: pathlib.Path = child / "images"
        flag_file: pathlib.Path = child / "extracted_images"
        if flag_file.is_file():
            logger.info("main - already processed %s images, skipping" % (child,))
            continue

        image_extractor: ImageExtractor = ImageExtractor(video_path, destination_path)
        image_extractor.extract_images()

        flag_file.touch()


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
