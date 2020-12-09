from typing import List
import logging
import pathlib
import shutil
import math
import operator

from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

logger = logging.getLogger("image_extractor")


class ImageExtractor:
    source_video_path: pathlib.Path
    destination_images_path: pathlib.Path

    def __init__(
        self, source_video_path: pathlib.Path, destination_images_path: pathlib.Path
    ):
        self.source_video_path = source_video_path
        self.destination_images_path = destination_images_path

    def diff_images(self, image1: np.ndarray, image2: np.ndarray) -> float:
        image1_gray: np.ndarray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray: np.ndarray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        diff: float = ssim(
            image1_gray, image2_gray, multichannel=False, gaussian_weights=True
        )
        return diff

    def get_images_from_video(
        self,
        vidcap: cv2.VideoCapture,
        start_milliseconds: int,
        length_milliseconds: int,
        interval_milliseconds: int = 500,
    ) -> List[np.ndarray]:

        max_count: int = math.floor(length_milliseconds / interval_milliseconds)
        count: int = 0
        return_value: List[np.ndarray] = []
        vidcap.set(cv2.CAP_PROP_POS_MSEC, start_milliseconds)
        while count < max_count:
            success, image = vidcap.read()
            return_value.append(image)
            count += 1

        return return_value

    def get_first_frame_from_video(self, vidcap: cv2.VideoCapture) -> np.ndarray:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, first_image = vidcap.read()
        return first_image

    def get_last_frame_from_video(self, vidcap: cv2.VideoCapture) -> np.ndarray:
        total_frames: int = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        success, last_image = vidcap.read()
        return last_image

    def get_duration_of_video(self, vidcap: cv2.VideoCapture) -> float:
        total_frames: int = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps: int = vidcap.get(cv2.CAP_PROP_FPS)
        return total_frames / fps

    def get_important_images(
        self, vidcap: cv2.VideoCapture, count: int, interval_milliseconds: int = 500
    ) -> List[np.ndarray]:
        logger.info(
            "get_important_images for video %s into %s"
            % (self.source_video_path, self.destination_images_path)
        )
        duration_seconds: float = self.get_duration_of_video(vidcap)
        max_chunk_size_miliseconds: int = math.floor(duration_seconds / count) * 1000
        last_frame: np.ndarray = self.get_last_frame_from_video(vidcap)
        return_value: List[np.ndarray] = []

        current_milliseconds: int = 0
        current_chunk_size_milliseconds: int = 1000
        while len(return_value) < (count - 1):
            logger.debug(
                "get_important_images - current_milliseconds %s"
                % (current_milliseconds,)
            )
            logger.debug("get_important_images - getting images..")
            current_images: List[np.ndarray] = self.get_images_from_video(
                vidcap,
                current_milliseconds,
                current_chunk_size_milliseconds,
                interval_milliseconds=interval_milliseconds,
            )
            logger.debug("get_important_images - calculating diffs...")
            if len(return_value) == 0:
                image_to_compare_to = last_frame
            else:
                image_to_compare_to = return_value[-1]
            diffs = [
                (self.diff_images(image, image_to_compare_to), image)
                for image in current_images
            ]
            diffs.sort(key=operator.itemgetter(0))
            logger.debug("get_important_images - found most important image in chunk")
            try:
                most_different_image: np.ndarray = diffs[0][1]
            except IndexError:
                logger.error(
                    "get_important_images failed for %s" % (self.source_video_path,)
                )
                most_different_image = current_images[0]
            return_value.append(most_different_image)

            current_milliseconds += current_chunk_size_milliseconds
            current_chunk_size_milliseconds = min(
                max_chunk_size_miliseconds, current_chunk_size_milliseconds * 2
            )

        return_value.append(last_frame)
        return return_value

    def extract_images(self) -> None:
        frames_path: pathlib.Path = self.destination_images_path
        logger.info("Extracting images into %s..." % (self.destination_images_path,))
        if frames_path.is_dir():
            shutil.rmtree(frames_path.absolute())
        frames_path.mkdir()

        vidcap = cv2.VideoCapture(str(self.source_video_path.absolute()))
        important_images: List[np.ndarray] = self.get_important_images(vidcap, count=5)
        for i, image in enumerate(important_images, start=1):
            filename: str = "frame%05d.jpg" % (i,)
            image_path: pathlib.Path = frames_path / filename
            cv2.imwrite(str(image_path.absolute()), image)
