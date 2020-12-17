from dataclasses import dataclass
from typing import List, Dict, Any, Iterator, Tuple, Optional
import cv2
import itertools
import json
import numpy as np
import pathlib
import pickle
import zstandard as zstd


class CameraEventNotFound(Exception):
    pass


@dataclass
class CameraEvent:
    root_path: pathlib.Path
    images_subpath: pathlib.Path
    augmented_images_subpath: pathlib.Path
    images_annotated_flag_file: pathlib.Path

    def load_precomputed_data(self) -> Tuple[List[List[np.array]], bool]:
        precomputed_data_path: pathlib.Path = self.get_precomputed_data_path()
        dctx = zstd.ZstdDecompressor()
        with precomputed_data_path.open("rb") as f_in:
            with dctx.stream_reader(f_in) as reader:
                unpickler = pickle.Unpickler(reader)
                data: Tuple[List[List[np.array]], bool] = unpickler.load()
                return data

    def save_precomputed_data(self, data: Tuple[List[List[np.array]], bool]) -> None:
        precomputed_data_path: pathlib.Path = self.get_precomputed_data_path()
        cctx = zstd.ZstdCompressor(level=3, threads=-1)
        with precomputed_data_path.open("wb") as f_out:
            with cctx.stream_writer(f_out) as compressor:
                pickler = pickle.Pickler(compressor, protocol=4)
                pickler.dump(data)
        self.get_precomputed_data_flag_path().touch()

    def get_precomputed_data_path(self) -> pathlib.Path:
        return self.root_path / "precomputed_data.pickle.zst"

    def get_precomputed_data_flag_path(self) -> pathlib.Path:
        return self.root_path / "precomputed_data_successful"

    def get_image_paths(self) -> List[pathlib.Path]:
        return sorted(filename for filename in self.images_subpath.iterdir())

    def get_augmented_image_paths(self) -> List[List[pathlib.Path]]:
        result: List[List[pathlib.Path]] = []
        children = sorted(child for child in self.augmented_images_subpath.iterdir())
        child: pathlib.Path
        for child in children:
            if not child.is_dir():
                continue
            subchildren: List[pathlib.Path] = sorted(
                subchild for subchild in child.iterdir()
            )
            result.append(subchildren)
        return result

    def get_event_info(self) -> Dict[str, Any]:
        data: Dict[str, Any]
        path: pathlib.Path = self.root_path / "event_info"
        with path.open() as f_in:
            data = json.load(f_in)
        return data

    def get_annotation_package_present(self) -> bool:
        data: Dict[str, Any]
        with self.images_annotated_flag_file.open() as f_in:
            data = json.load(f_in)
        return data["package_present"] is True

    def get_camera_event_actual_images_as_numpy_data(
        self, image_x_pixels: int = 299, image_y_pixels: int = 299
    ) -> Tuple[List[np.array], Optional[bool]]:
        imgs: List[np.array] = []
        for image in self.get_image_paths():
            img: np.array = cv2.imread(str(image.absolute()))
            img = cv2.resize(
                img, (image_x_pixels, image_y_pixels), interpolation=cv2.INTER_LANCZOS4,
            )
            imgs.append(img)

        annotation: Optional[bool]
        if self.images_annotated_flag_file.is_file():
            annotation = self.get_annotation_package_present()
        else:
            annotation = None
        return (imgs, annotation)

    def get_camera_event_package_present_as_numpy_data(
        self, image_x_pixels: int = 299, image_y_pixels: int = 299
    ) -> Tuple[List[List[np.array]], bool]:
        """Convert camera event into (augmented images, boolean for is package is present).

        Augmented images is [[image_1, image_2, image_3, ...]].
        """
        augmented_images: List[List[np.array]] = []
        augmented_image_chunk: List[pathlib.Path]
        for augmented_image_chunk in self.get_augmented_image_paths():
            imgs: List[np.array] = []
            for augmented_image in augmented_image_chunk:
                img: np.array = cv2.imread(str(augmented_image.absolute()))
                img = cv2.resize(
                    img,
                    (image_x_pixels, image_y_pixels),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                imgs.append(img)
            augmented_images.append(imgs)

        return (augmented_images, self.get_annotation_package_present())

    def annotate_package_present(self) -> None:
        print("annotation %s as package present" % (self.root_path,))
        data: Dict[str, Any] = {
            "images_subpath": self.images_subpath.stem,
            "package_present": True,
        }
        with self.images_annotated_flag_file.open("w") as f_out:
            json.dump(data, f_out)

    def annotate_package_not_present(self) -> None:
        print("annotation %s as package not present" % (self.root_path,))
        data: Dict[str, Any] = {
            "images_subpath": self.images_subpath.stem,
            "package_present": False,
        }
        with self.images_annotated_flag_file.open("w") as f_out:
            json.dump(data, f_out)


class CameraEventManager:
    root_path: pathlib.Path
    images_subpath: str

    def __init__(
        self, root_path: pathlib.Path, images_subpath: str = "images",
    ) -> None:
        self.root_path = root_path
        self.images_subpath = images_subpath

    def get_unannotated_events(self) -> Iterator[CameraEvent]:
        return self.get_events(get_annotated_events=False)

    def get_annotated_events(self) -> Iterator[CameraEvent]:
        return self.get_events(get_annotated_events=True)

    def get_all_events(self) -> Iterator[CameraEvent]:
        return itertools.chain(
            self.get_annotated_events(), self.get_unannotated_events()
        )

    def get_event(self, event_id: str) -> CameraEvent:
        path = self.root_path / event_id
        if not path.is_dir():
            raise CameraEventNotFound()
        images_path: pathlib.Path = path / self.images_subpath
        augmented_images_path: pathlib.Path = path / f"{self.images_subpath}_augmented"
        flag_filename: str = "%s_annotated" % (self.images_subpath,)
        images_annotated_flag_file: pathlib.Path = path / flag_filename
        return CameraEvent(
            root_path=path,
            images_subpath=images_path,
            augmented_images_subpath=augmented_images_path,
            images_annotated_flag_file=images_annotated_flag_file,
        )

    def get_events(self, get_annotated_events: bool) -> Iterator[CameraEvent]:
        paths: List[pathlib.Path] = [path for path in self.root_path.iterdir()]
        paths.sort()

        child: pathlib.Path
        for child in paths:
            if not child.is_dir():
                continue
            images_path: pathlib.Path = child / self.images_subpath
            if not images_path.is_dir():
                continue
            augmented_images_path: pathlib.Path = child / f"{self.images_subpath}_augmented"
            flag_filename: str = "%s_annotated" % (self.images_subpath,)
            images_annotated_flag_file: pathlib.Path = child / flag_filename
            if get_annotated_events is False and images_annotated_flag_file.is_file():
                print("%s is already annotated, skipping..." % (child,))
                continue
            if (
                get_annotated_events is True
                and not images_annotated_flag_file.is_file()
            ):
                print("%s is not annotated, skipping..." % (child,))
                continue
            yield CameraEvent(
                root_path=child,
                images_subpath=images_path,
                augmented_images_subpath=augmented_images_path,
                images_annotated_flag_file=images_annotated_flag_file,
            )
